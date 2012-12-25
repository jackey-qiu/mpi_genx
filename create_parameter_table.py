import numpy as np
#this class is used to create table quickly to be imported in GenX
'''raw match file looks like
Four segments are seperated by ";", the last subsegment is seperated by "_" with all the other segements are seperated by "'"
gp_O1O8.,gp_O2O7.,gp_Fe2Fe9.,gp_Fe3Fe8.,gp_O3O10.,gp_O4O9.,gp_Fe4Fe12.,gp_Fe6Fe10.,gp_O5O12.,gp_O6O11.;setdx,setdy,setdz;True;[-0.2,0.2]_[-0.2,0.2]_[-0.2,0.2]
domain1A.;setFe1_2_0m,setFe1_3_0m;False;[0,0]_[0,0]
'''
class create_table:
    def __init__(self,raw_match_file='F:\\',save_file_path=''):
        self.f1=open(raw_match_file)
        self.f2=open(save_file_path,'w')
        self.create_tab()
    
    def create_tab(self):
        lines=self.f1.readlines()
        for line in lines:
            if line[0]!='#':
                segment=line.strip().rsplit(';')
                segment[0]=segment[0].rsplit(',')
                segment[1]=segment[1].rsplit(',')
                segment[3]=segment[3].rsplit('_')
                for i in range(len(segment[0])):
                    for j in range(len(segment[1])):
                        #print list(segment[3][j])
                        self.f2.write("%s%s\t%5.3e\t%s\t%5.3e\t%5.3e\t%s\t\n"%(segment[0][i],segment[1][j],np.mean(eval(segment[3][j])),segment[2],eval(segment[3][j])[0],eval(segment[3][j])[1],'-'))
        self.f1.close()
        self.f2.close()

if __name__=='__main__':
    test1=create_parameter_table.create_table('raw match file.txt','table file.txt')
    test1.create_tab()
    