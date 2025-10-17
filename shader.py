


vertex_shader= """
    #version 330 core
    layout(location=0) in vec3 aPosition;
    layout(location=1) in float aScalar;

    uniform mat4 m_model;
    uniform mat4 m_view;
    uniform mat4 m_proj;

    out VS_OUT {
        float s;
        vec3  posWorld;       // pass WORLD-SPACE position
    } vs;

    void main() {
        vec4 world = m_model * vec4(aPosition, 1.0);
        vs.posWorld = world.xyz;
        gl_Position = m_proj * (m_view * world);
        vs.s = aScalar;
    }
    """


fragment_shader= """
    #version 330 core
    in VS_OUT {
        float s;
        vec3  posWorld;        // WORLD-SPACE position
    } fs;

    out vec4 fragColor;

    // scalar → color controls
    uniform float uSMin;
    uniform float uSMax;
    uniform float uGamma;
    uniform sampler2D uLUT;    // 1D LUT (height=1)
    uniform int   uNumColors;  // discrete bands count

    // POINT LIGHT in WORLD SPACE
    uniform vec3 uLightPosWorld;   // light position (world)
    uniform vec3 uCameraPosWorld;  // camera/eye position (world)

    // lighting params
    uniform vec3  uAmbient;        // e.g. 0.08
    uniform vec3  uSpecColor;      // e.g. white
    uniform float uShininess;      // e.g. 64

    void main() {
        // Face normal in WORLD space from screen-space derivatives of WORLD pos
        vec3 dx = dFdx(fs.posWorld);
        vec3 dy = dFdy(fs.posWorld);
        vec3 N  = normalize(cross(dx, dy));
        N *= (gl_FrontFacing ? 1.0 : -1.0);

        // Vectors (WORLD space)
        vec3 L = normalize(uLightPosWorld - fs.posWorld);   // to light
        vec3 V = normalize(uCameraPosWorld - fs.posWorld);  // to camera

        // Diffuse term
        float NdotL = max(dot(N, L), 0.0);

        // Blinn-Phong specular
        vec3  H     = normalize(L + V);
        float NdotH = max(dot(N, H), 0.0);
        float spec  = (NdotL > 0.0) ? pow(NdotH, max(uShininess, 1.0)) : 0.0;

        // Scalar normalization
        float s = fs.s;
        if (isnan(s)) {
            // Example: vivid magenta (typical "invalid data" color)
            fragColor = vec4(1.0, 0.0, 1.0, 1.0);
            return;
        }
        float t = (fs.s - uSMin) / max(uSMax - uSMin, 1e-8);
        //t = clamp(t, 0.0, 1.0);
        t = pow(t, max(uGamma, 1e-2));

        // --- Discretize into bins
        float bins = max(float(uNumColors), 1.0); // avoid /0
        float tb   = t * bins;
        float k    = clamp(floor(tb), 0.0, bins - 1.0);
        float tc   = (k + 0.5) / bins;

        // --- Base color from LUT (sample at row 0.5)
        vec3 base = texture(uLUT, vec2(tc, 0.5)).rgb;

        // --- Lighting
        vec3 lit = base * (uAmbient + NdotL) + uSpecColor * spec;

        // --- Black borders at bin edges
        float binPos   = fract(tb);
        float borderFrac = 0.0005;          // ~2% of a bin
        float px       = fwidth(tb);      // resolution-aware thickness
        float edgeDist = min(binPos, 1.0 - binPos);

        // AA borders (use step(edgeDist, borderFrac) for hard edges)
        float edgeMask = 1.0 - smoothstep(borderFrac, borderFrac + px, edgeDist);
        edgeMask = clamp(edgeMask, 0.0, 1.0);

        // Apply border last (override lighting)
        vec3 color = mix(lit, vec3(0)/256, edgeMask);

        fragColor = vec4(color, 1.0);
    }
    """
