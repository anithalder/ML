import pandas as pd 
class CSV:
    def __init__(self,filename):
        self.filename=filename
        self.x=[]
        self.y=[]
        self.data=None
    def readFile(self):
        self.data=pd.read_csv(self.filename)
        print(self.data.head())
    def getdata(self):
        self.x=self.data['Score'].tolist()
        self.y=self.data['Pass/Fail'].tolist()
        return self.x ,self.y