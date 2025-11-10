
import numpy as np
import pandas as pd
import traceback
import json

class LAT_points:
    def __init__(self,jason_list:json):
        self.jason=jason_list
        self.Points:list[point] = []
        self.extract()


    def extract(self):
        self.First=[]
        self.Second=[]
        self.Third=[]
        self.SR=[]
        self.labels=[]
        self.p_numbers=[]
        self.Voltage_sinus=[]

        self.Third_Delta=[]
        self.Third_dur=[]
        self.Third_deflection=[]
        self.Third_Voltage=[]

        self.First_Delta=[]
        self.First_dur=[]
        self.First_deflection=[]
        self.First_Voltage=[]

        self.Second_Delta=[]
        self.Second_dur=[]
        self.Second_deflection=[]
        self.Second_Voltage=[]

        self.min_Voltage=[]

        self.Sinus_dur=[]
        for p in self.jason:
            new_point,flag=point.init(p)

            # LATs
            if not np.any(np.isin(["Pass","No_label","No_response"],flag)):
                self.Points.append(new_point)
                self.First.append(new_point.First)
                self.Second.append(new_point.Second)
                self.Third.append(new_point.Third)
                self.SR.append(new_point.SR)

                #Labels
                self.labels.append(new_point.label)
                self.p_numbers.append(new_point.point_number)
                
                #SR
                self.Sinus_dur.append(new_point.Sinus_dur)
                self.Voltage_sinus.append(new_point.Voltage_sinus)
                
                #Third Stim
                self.Third_Delta.append(new_point.Third_delta)
                self.Third_dur.append(new_point.Third_dur)
                self.Third_deflection.append(new_point.Third_deflection)
                self.Third_Voltage.append(new_point.Third_voltage)

                #First Stim
                self.First_Delta.append(new_point.First_delta)
                self.First_dur.append(new_point.First_dur)
                self.First_deflection.append(new_point.First_deflection)
                self.First_Voltage.append(new_point.First_voltage)

                #Second Stim
                self.Second_Delta.append(new_point.Second_delta)
                self.Second_dur.append(new_point.Second_dur)
                self.Second_deflection.append(new_point.Second_deflection)
                self.Second_Voltage.append(new_point.Second_voltage)

                self.min_Voltage.append(new_point.voltage)

            
        self.data=pd.DataFrame(np.array([self.First,self.Second,self.Third,self.SR,self.labels,self.p_numbers]).T,columns=
                               ["First","Second","Third","SR","label","point_number"])
        return self
    




class point:
    def __init__(self,list_data:list):
        self.label:str
        self.point_number:str
        self.lats:list

        self.First,self.Second,self.Third,self.SR,self.Sinus_dur,self.First_dur,self.Second_dur,self.Third_dur=[None]*8
        self.First_delta,self.Second_delta,self.Third_delta,self.First_deflection,self.Second_deflection,self.Third_deflection=[None]*6
        self.First_voltage,self.Second_voltage,self.Third_voltage,self.voltage,self.Voltage_sinus=[None]*5
    @classmethod
    def init(cls,list_data:list):
        flag=[]
        self=cls(list_data)
        try:
            self.label=list_data[1]
            self.point_number=list_data[0]
            self.lats=list_data[2]
        except:
            flag.append("Pass")
            return self,flag
        lats=self.lats
        
        if "sinus" in lats.keys() and lats["sinus"]:
            if len(lats["sinus"])==0:
                self.label=None
                print(f"No Sinus detected point number: {self.point_number}")
                flag.append("Pass")
                return self,flag
            else:
                self.SR=lats["sinus"][0][0]-lats["refs_sinus"][0]
        if "stim" in lats.keys() and lats["stim"]:
            if len(lats["stim"])==3:
                self.First=lats["stim"][0][0]-lats["refs_stim"][0]
                self.Second=lats["stim"][1][0]-lats["refs_stim"][1]
                self.Third=lats["stim"][2][0]-lats["refs_stim"][2]
            else:
                flag.append("Pass")
                return self,flag

        



            index=None
            for i,sig in enumerate(lats["sinus"]):
                if sig[0]>lats["stim"][0][0]:
                    index=i-1
                
            if index is None:
                index=-1

            self.Sinus_dur=(lats["sinus"][index][1]-lats["sinus"][index][0])
            self.Voltage_sinus=lats["voltage_sinus"][index]
            self.Third_dur=lats["stim"][2][1]-lats["stim"][2][0]
            self.First_dur=lats["stim"][0][1]-lats["stim"][0][0]
            self.Second_dur=lats["stim"][1][1]-lats["stim"][1][0]

                
            self.Third_delta=self.Third_dur-self.Sinus_dur
            self.First_delta=self.First_dur-self.Sinus_dur
            self.Second_delta=self.Second_dur-self.Sinus_dur
                
            if 'deflection_stim' in lats.keys() and lats['deflection_stim']:
                self.Third_deflection=lats['deflection_stim'][2]
                self.First_deflection=lats['deflection_stim'][0]
                self.Second_deflection=lats['deflection_stim'][1]
            else:
                flag.append("No_deflection")


            if "voltage_stim" in lats.keys() and lats["voltage_stim"]:
                self.Third_voltage=lats["voltage_stim"][2]
                self.First_voltage=lats["voltage_stim"][0]
                self.Second_voltage=lats["voltage_stim"][1]
                self.voltage=np.min([self.First_voltage,self.Second_voltage,self.Third_voltage]) 

            else:
                flag.append("No_voltage")


            if self.voltage<0.07 or self.label.lower()=="reject":
                flag.append("No_response")
                return self,flag 
            
            label=self.label
            if np.isin(label.lower(),["","-","negative","neg","naranja","naran","nar"]):
                self.label="NEG"
            elif np.isin(label.lower(),["+","pos","positive","verde","ver","green"]) or "+" in label:
                self.label="POS"
            else:
                flag.append("No_label")
                return self,flag 
            return self,flag
