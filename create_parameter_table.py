import numpy as np
#this class is used to create table quickly to be imported in GenX
#INTERFACE_ATOMS_1:atom ids at the interface before binding sorbates
#value_match:define the range of different variables
INTERFACE_ATOMS_1=['Fe1_4_0','Fe1_6_0','O1_1_0','O1_2_0','O1_3_0','O1_4_0','O1_5_0','O1_6_0']
value_match={('Fe','dx1'):[-0.05,0.05],('Fe','dy1'):[-0.05,0.05],('Fe','dz1'):[-0.05,0.05],('Fe','m'):[0.,1.],('Fe','u'):[0.3,0.6],('Fe','oc'):[0.,1.],\
              ('O1','dx1'):[-0.05,0.05],('O1','dy1'):[-0.05,0.05],('O1','dz1'):[-0.05,0.05],('O1','u'):[0.3,0.8],('O1','oc'):[0.,1.],\
              ('HO','dx1'):[-1.,1.],('HO','dy1'):[-1.,1.],('HO','dz1'):[-0.5,0.5],('HO','u'):[0.4,10.],('HO','oc'):[0.,1.],\
              ('Pb','dx1'):[-1.,1.],('Pb','dy1'):[-1.,1.],('Pb','dz1'):[-0.5,0.5],('Pb','u'):[0.,0.7],('Pb','oc'):[0.,1.]}
class create_table:
    def __init__(self,filepath='F:\\',sorbate_tag=[[1,0]],heads=['domain1A.set'],heads_match1=['domain1A.set'],interfacial_atoms=[INTERFACE_ATOMS_1]):
        #sorbate_tag=[[1,0],[2,3]]:two domains with first domain having 1 pb and 0 Oxygen, second 2 pb's and 3 Oxygens
        #heads is the front segment of match0, heads_match1 is the front segment of match1
        filename='file'
        if len(sorbate_tag)==2:
            filename='double_domain_'+str(sorbate_tag[0][0])+'pb_'+str(sorbate_tag[0][1])+'O_'+str(sorbate_tag[1][0])+'pb_'+str(sorbate_tag[1][1])+'O'
        elif len(sorbate_tag)==1:
            filename='single_domain_'+str(sorbate_tag[0][0])+'pb_'+str(sorbate_tag[0][1])+'O'
        self.f=open(filepath+filename,'w')
        self.sorbate_tag=sorbate_tag
        self.interfacial_atoms=interfacial_atoms
        self.match0=[heads,interfacial_atoms,['dx1','dy1','dz1','u','oc']]
        self.match1=[heads_match1,['Fe1_2_0','Fe1_3_0'],['m']]
        self.match2=[['inst.','rgh.'],['set_inten','setBeta'],[[1.,4.],[0,0.5]]]
        self.create_match0()
    
    def create_match0(self):
        #print len(self.sorbate_tag)
        for i in range(len(self.sorbate_tag)):
            for j in range(self.sorbate_tag[i][0]):
                #print range(self.sorbate_tag[i][0]),'Pb'+str(j+1)
                self.match0[1][i].append('Pb'+str(j+1))
            if (self.sorbate_tag[i][1]!=0):
                #print self.sorbate_tag[i][1]
                for k in range(self.sorbate_tag[i][1]):
                    self.match0[1][i].append('HO_'+str(k+1))
            
    def case_match0(self,match0):
        for i in match0[0]:
            for j in match0[1]:
                for k in j:
                    for m in match0[2]:
                        self.f.write("%s%s%s\t%5.6e\t%s\t%5.6e\t%5.6e\t%s\t\n"%(i,k,m,value_match[(k[0:2],m)][0],'True',value_match[(k[0:2],m)][0],value_match[(k[0:2],m)][1],'-'))
    
    def case_match1(self,match1):
        for i in match1[0]:
            for j in match1[1]:
                for k in match1[2]:
                    self.f.write("%s%s%s\t%5.6e\t%s\t%5.6e\t%5.6e\t%s\t\n"%(i,j,k,value_match[(j[0:2],k)][0],'False',value_match[(j[0:2],k)][0],value_match[(j[0:2],k)][1],'-'))
    
    def case_match2(self,match2):
        for i in range(len(match2[0])):
            self.f.write("%s%s\t%5.6e\t%s\t%5.6e\t%5.6e\t%s\t\n"%(match2[0][i],match2[1][i],match2[2][i][0],'True',match2[2][i][0],match2[2][i][1],'-'))
    
    def action(self):
        self.case_match0(self.match0)
        self.case_match1(self.match1)
        self.case_match2(self.match2)
        self.f.close()

        
if __name__=='__main__':
    test1=create_parameter_table.create_table('table file.txt')
    test1.action()