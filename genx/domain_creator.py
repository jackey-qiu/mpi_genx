# -*- coding: utf-8 -*-
import models.sxrd_test5_sym_new_test_new66_2_2 as model
from models.utils import UserVars
import numpy as np
from operator import mul
from numpy.linalg import inv
from random import uniform
import sys
sys.path.append('D:\\Programming codes\\geometry codes\\polyhedra-geometry')
import hexahedra,hexahedra_distortion,tetrahedra,octahedra,tetrahedra_edge_distortion,trigonal_pyramid_distortion,trigonal_pyramid_distortion_shareface,trigonal_pyramid_distortion2,trigonal_pyramid_distortion3,trigonal_pyramid_distortion4
import trigonal_pyramid_known_apex
from domain_creator_water import domain_creator_water
from domain_creator_sorbate import domain_creator_sorbate
from domain_creator_surface import domain_creator_surface

"""functions in this class
create_grid_number, compare_grid, create_match_lib: supporting functions for the bv calculation
find_neighbors: find neighbors of a specific atoms within a range, retrun atm_ids and offset
cal_bond_valence1: cal bv of an atom and neiboring atoms within some range, will return a lib with keys of the id of neibor atoms and "total_valence"
cal_bond_valence2: cal bv based on a match_list of form like [['Fe1','Fe2'],['-x','+y']], which specifys the neibor atoms, will return a 
                   bond_valence_container with keys from the first item of the list and "total" representing the total bv
cal_bond_valence3: cal bv based on match_lib of form like {'O1':[['Fe1','Fe2'],['-x','+y']]}, will return a lib with the same keys and the values of 
                   each key the associated bv calculated

create_coor_transformation: create a spherical coor frame with three atms (z vec normal to the plane), return transformation matrix T (last col define the origin coordinates)
extract_spherical_pars: cal the r theta and phi in the spherical frame for a specific atom 
set_sorbate_xyz: with known r theta and phi values, cal the associated xyz in the original coor frame and set the coor as new pst of an atom in the domain

scale_opt_batch:scale fitting parameters towards deeper layer, it takes a filename as argument
set_new_vars: set new variables in a sequence, like u_1,u_2,u_3
set_discrete_new_vars_batch:set discrete new variables, take a filename as argument 
init_sim_batch: update the containor of free varibles, eg  u_containor=[u1,u2,u3,u4,u5], during fitting u will be updated, excute this function to update u_containor
"""

x0_v,y0_v,z0_v=np.array([1.,0.,0.]),np.array([0.,1.,0.]),np.array([0.,0.,1.])

#anonymous function f1 calculating transforming matrix with the basis vector expressions,x1y1z1 is the original basis vector
#x2y2z2 are basis of new coor defined in the original frame,new=T.orig
f1=lambda x1,y1,z1,x2,y2,z2:np.array([[np.dot(x2,x1),np.dot(x2,y1),np.dot(x2,z1)],\
                                      [np.dot(y2,x1),np.dot(y2,y1),np.dot(y2,z1)],\
                                      [np.dot(z2,x1),np.dot(z2,y1),np.dot(z2,z1)]])

#f2 calculate the distance b/ p1 and p2
f2=lambda p1,p2:np.sqrt(np.sum((p1-p2)**2))

#anonymous function f3 is to calculate the coordinates of basis with magnitude of 1.,p1 and p2 are coordinates for two known points, the 
#direction of the basis is pointing from p1 to p2
f3=lambda p1,p2:(1./f2(p1,p2))*(p2-p1)+p1

#extract xyz for atom with id in domain
def extract_coor(domain,id):
    index=np.where(domain.id==id)[0][0]
    x=domain.x[index]+domain.dx1[index]+domain.dx2[index]+domain.dx3[index]
    y=domain.y[index]+domain.dy1[index]+domain.dy2[index]+domain.dy3[index]
    z=domain.z[index]+domain.dz1[index]+domain.dz2[index]+domain.dz3[index]
    return np.array([x,y,z])

def extract_component(domain,id,name_list):
    index=np.where(domain.id==id)[0][0]
    temp=[vars(domain)[name][index] for name in name_list]
    for i in range(len(name_list)):
        print name_list[i]+'=',temp[i]
        
#set coor to atom with id in domain
def set_coor(domain,id,coor):
    index=np.where(domain.id==id)[0][0]
    domain.x[index]=coor[0]
    domain.y[index]=coor[1]
    domain.z[index]=coor[2]
    
#set sorbate coors to two symmetrical realated domains (rcut hematite in this case)
#domains:two items with symmetry related to each other
#ids:two items, first item is a list of ids corresponding to first domain, and second item to the second domain
#els:a list of elements with order corresponding to each item of the ids
#grids: fractional coordinates of sorbates for first domain
def set_coor_grid(domains=['domain1A','domain1B'],ids=[[],[]],els=[],grids=[[0.3,0.5,1.2]],random_switch=False):
    grids=grids
    if random_switch==True:
        grids=[[uniform(0,1),uniform(0,1),uniform(2,2.5)]for el in els]
    for i in range(len(els)):
        index=None
        try:
            index=np.where(domains[0].id==ids[0][i])[0][0]
        except:
            domains[0].add_atom( ids[0][i], els[i],  grids[i][0] ,grids[i][1], grids[i][2] ,0.5,     1.00000e+00 ,     1.00000e+00 )
            domains[1].add_atom( ids[1][i], els[i],  1-grids[i][0] ,grids[i][1]-0.06955, grids[i][2]-0.5 ,0.5,     1.00000e+00 ,     1.00000e+00 )
        if index!=None:
            domain[0].x[index],domain[0].y[index],domain[0].z[index]=grids[i][0],grids[i][1],grids[i][2]
            domain[1].x[index],domain[1].y[index],domain[1].z[index]=1-grids[i][0],grids[i][1]-0.06955,grids[i][2]-0.5
#grid matching library for considering offset, x y both from -0.3 to 1.2 with each step of 0.5
#match like 1  2  3
#           6  5  4
#           7  8  9
#the match is based on closest distance
#if you consider match 3 and 6, then 6 will shift towards right by 1 unit to make it to be adjacent to 3, so in this case offset is "+y"
#5 is neighbor to all the other tiles so no offsets (depicted as None)
grid_match_lib={}
grid_match_lib[1]={2:None,3:'-x',4:'-x',5:None,6:None,7:'+y',8:'+y',9:'-x+y'}
grid_match_lib[2]={1:None,3:None,4:None,5:None,6:None,7:'+y',8:'+y',9:'+y'}
grid_match_lib[3]={2:None,1:'+x',4:None,5:None,6:'+x',7:'+x+y',8:'+y',9:'+y'}
grid_match_lib[4]={2:None,3:None,1:'+x',5:None,6:'+x',7:'+x',8:None,9:None}
grid_match_lib[5]={2:None,3:None,4:None,1:None,6:None,7:None,8:None,9:None}
grid_match_lib[6]={2:None,3:'-x',4:'-x',5:None,1:None,7:None,8:None,9:'-x'}
grid_match_lib[7]={2:'-y',3:'-x-y',4:'-x',5:None,6:None,1:'-y',8:None,9:'-x'}
grid_match_lib[8]={2:'-y',3:'-y',4:None,5:None,6:None,7:None,1:'-y',9:None}
grid_match_lib[9]={2:'-y',3:'-y',4:None,5:None,6:'+x',7:'+x',8:None,1:'+x-y'}

################################some functions to be called in GenX script#######################################
#atoms (sorbates) will be added to position specified by the coor(usually set the coor to the center, then you can easily set dxdy range to [-0.5,0.5] [
def add_atom(domain,ref_coor=[],ids=[],els=[]):
    for i in range(len(ids)):
        try:
            domain.add_atom(ids[i],els[i],ref_coor[i][0],ref_coor[i][1],ref_coor[i][2],0.5,1.0,1.0)
        except:
            index=np.where(domain.id==ids[i])[0][0]
            domain.x[index]=ref_coor[i][0]
            domain.y[index]=ref_coor[i][1]
            domain.z[index]=ref_coor[i][2]

#function to export refined atoms positions after fitting
def print_data(N_sorbate=4,N_atm=40,domain='',z_shift=1,save_file='D://model.xyz'):
    data=domain._extract_values()
    index_all=range(len(data[0]))
    index=index_all[0:20]+index_all[N_atm:N_atm+N_sorbate]
    f=open(save_file,'w')
    for i in index:
        s = '%-5s   %7.5e   %7.5e   %7.5e\n' % (data[3][i],data[0][i]*5.038,(data[1][i]-0.1391)*5.434,(data[2][i]-z_shift)*7.3707)
        f.write(s)
    f.close()

#function to export ref fit file (a connection between GenX output and ROD input)
#NOTE:only print out surface atoms, and no sorbates
def print_data_for_ROD(N_atm=40,domain='',save_file='D:\\Google Drive\\useful codes\\half_layer_GenX_s1.txt'):
    data=domain._extract_values()
    index_all=range(len(data[0]))
    index=index_all[0:40]
    f=open(save_file,'w')
    f2=open(save_file[:-6]+'s2.txt','w')
    for i in index:
        s = '%s\t%6.5f\t%i\t%i\t%i\t%i\t%6.5f\t%i\t%i\t%i\t%i\t%6.5f\t%i\t%i\t%i\n' % (data[3][i],data[0][i],1,0,0,0,data[1][i],1,0,0,0,data[2][i],0,0,0)
        f.write(s)
    for i in index_all[0:30]:
        s = '%s\t%6.5f\t%i\t%i\t%i\t%i\t%6.5f\t%i\t%i\t%i\t%i\t%6.5f\t%i\t%i\t%i\n' % (data[3][i],1.-data[0][i],-1,0,0,0,data[1][i]-0.1391/2.,1,0,0,0,data[2][i]-0.5,0,0,0)
        f2.write(s)
    f.close()
    f2.close()
    
def create_list(ids,off_set_begin,start_N):
    ids_processed=[[],[]]
    off_set=[None,'+x','-x','+y','-y','+x+y','+x-y','-x+y','-x-y']
    for i in range(start_N):
        ids_processed[0].append(ids[i])
        ids_processed[1].append(off_set_begin[i])
    for i in range(start_N,len(ids)):
        for j in range(9):
            ids_processed[0].append(ids[i])
            ids_processed[1].append(off_set[j])
    return ids_processed 
#function to build reference bulk and surface slab  
def add_atom_in_slab(slab,filename):
    f=open(filename)
    lines=f.readlines()
    for line in lines:
        if line[0]!='#':
            items=line.strip().rsplit(',')
            slab.add_atom(str(items[0]),str(items[1]),float(items[2]),float(items[3]),float(items[4]),float(items[5]),float(items[6]),float(items[7]))

#here only consider the match in the bulk,the offset should be maintained during fitting
def create_match_lib_before_fitting(domain_class,domain,atm_list,search_range):
    match_lib={}
    for i in atm_list:
        atms,offset=domain_class.find_neighbors(domain,i,search_range)
        match_lib[i]=[atms,offset]
    return match_lib
#Here we consider match with sorbate atoms, sorbates move around within one unitcell, so the offset will be change accordingly
#So this function should be placed inside sim function
#Note there is no return in this function, which will only update match_lib
def create_match_lib_during_fitting(domain_class,domain,atm_list,pb_list,HO_list,match_lib):
    match_list=[[atm_list,pb_list+HO_list],[pb_list,atm_list+HO_list],[HO_list,atm_list+pb_list]]
    #[atm_list,pb_list+HO_list]:atoms in atm_list will be matched to atoms in pb_list+HO_list
    for i in range(len(match_list)):
        atm_list_1,atm_list_2=match_list[i][0],match_list[i][1]
        for atm1 in atm_list_1:
            grid=domain_class.create_grid_number(atm1,domain)
            for atm2 in atm_list_2:
                grid_compared=domain_class.create_grid_number(atm2,domain)
                offset=domain_class.compare_grid(grid,grid_compared)
                if atm1 in match_lib.keys():
                    match_lib[atm1][0].append(atm2)
                    match_lib[atm1][1].append(offset)
                else:
                    match_lib[atm1]=[[atm2],[offset]]

def create_sorbate_ids(el='Pb',N=2,tag='_D1A'):
    id_list=[]
    [id_list.append(el+str(i+1)+tag) for i in range(N)]
    return id_list
    
class domain_creator(domain_creator_water,domain_creator_sorbate,domain_creator_surface):
    def __init__(self,ref_domain,id_list,terminated_layer=0,domain_tag='_D1',new_var_module=None):
        #id_list is a list of id in the order of ref_domain,terminated_layer is the index number of layer to be considered
        #for termination,domain_N is a index number for this specific domain, new_var_module is a UserVars module to be used in
        #function of set_new_vars
        self.ref_domain=ref_domain
        self.id_list=id_list
        self.terminated_layer=terminated_layer
        self.domain_tag=domain_tag
        self.share_face,self.share_edge,self.share_corner=(False,False,False)
        #self.anchor_list=[]
        self.polyhedra_list=[]
        self.new_var_module=new_var_module
        self.domain_A,self.domain_B=self.create_equivalent_domains_2()
    
    def build_super_cell(self,ref_domain,rem_atom_ids=None):
    #build a super cell based on the ref_domain, the super cell is actually two domains stacking together in x direction
    #rem_atom_ids is a list of atom ids you want to remove before building a super cell
        super_cell=ref_domain.copy()
        if rem_atom_ids!=None:
            for i in rem_atom_ids:
                super_cell.del_atom(i)
                
        def _extract_coor(domain,id):
            index=np.where(domain.id==id)[0][0]
            x=domain.x[index]+domain.dx1[index]+domain.dx2[index]+domain.dx3[index]
            y=domain.y[index]+domain.dy1[index]+domain.dy2[index]+domain.dy3[index]
            z=domain.z[index]+domain.dz1[index]+domain.dz2[index]+domain.dz3[index]
            return np.array([x,y,z])
            
        for id in super_cell.id:
            index=np.where(ref_domain.id==id)[0][0]
            super_cell.add_atom(id=str(id)+'_+x',element=ref_domain.el[index], x=_extract_coor(ref_domain,id)[0]+1.0, y=_extract_coor(ref_domain,id)[1], z=_extract_coor(ref_domain,id)[2], u = ref_domain.u[index], oc = ref_domain.oc[index], m = ref_domain.m[index])
            super_cell.add_atom(id=str(id)+'_-x',element=ref_domain.el[index], x=_extract_coor(ref_domain,id)[0]-1.0, y=_extract_coor(ref_domain,id)[1], z=_extract_coor(ref_domain,id)[2], u = ref_domain.u[index], oc = ref_domain.oc[index], m = ref_domain.m[index])
            super_cell.add_atom(id=str(id)+'_+y',element=ref_domain.el[index], x=_extract_coor(ref_domain,id)[0], y=_extract_coor(ref_domain,id)[1]+1., z=_extract_coor(ref_domain,id)[2], u = ref_domain.u[index], oc = ref_domain.oc[index], m = ref_domain.m[index])
            super_cell.add_atom(id=str(id)+'_-y',element=ref_domain.el[index], x=_extract_coor(ref_domain,id)[0], y=_extract_coor(ref_domain,id)[1]-1., z=_extract_coor(ref_domain,id)[2], u = ref_domain.u[index], oc = ref_domain.oc[index], m = ref_domain.m[index])
            super_cell.add_atom(id=str(id)+'_+x-y',element=ref_domain.el[index], x=_extract_coor(ref_domain,id)[0]+1.0, y=_extract_coor(ref_domain,id)[1]-1., z=_extract_coor(ref_domain,id)[2], u = ref_domain.u[index], oc = ref_domain.oc[index], m = ref_domain.m[index])
            super_cell.add_atom(id=str(id)+'_+x+y',element=ref_domain.el[index], x=_extract_coor(ref_domain,id)[0]+1.0, y=_extract_coor(ref_domain,id)[1]+1., z=_extract_coor(ref_domain,id)[2], u = ref_domain.u[index], oc = ref_domain.oc[index], m = ref_domain.m[index])
            super_cell.add_atom(id=str(id)+'_-x+y',element=ref_domain.el[index], x=_extract_coor(ref_domain,id)[0]-1.0, y=_extract_coor(ref_domain,id)[1]+1., z=_extract_coor(ref_domain,id)[2], u = ref_domain.u[index], oc = ref_domain.oc[index], m = ref_domain.m[index])
            super_cell.add_atom(id=str(id)+'_-x-y',element=ref_domain.el[index], x=_extract_coor(ref_domain,id)[0]-1.0, y=_extract_coor(ref_domain,id)[1]-1., z=_extract_coor(ref_domain,id)[2], u = ref_domain.u[index], oc = ref_domain.oc[index], m = ref_domain.m[index])
        
        return super_cell
    
    def create_equivalent_domains(self):
        new_domain_A=self.ref_domain.copy()
        new_domain_B=self.ref_domain.copy()
        for id in self.id_list[:self.terminated_layer*2]:
            if id!=[]:
                new_domain_A.del_atom(id)
        #number 5 here is crystal specific, here is the case for hematite
        for id in self.id_list[:(self.terminated_layer+5)*2]:
            #print id in new_domain_B.id
            new_domain_B.del_atom(id)
        return new_domain_A,new_domain_B

    def create_equivalent_domains_2(self):
        new_domain_A=self.ref_domain.copy()
        new_domain_B=self.ref_domain.copy()
        for id in self.id_list[:self.terminated_layer*2]:
            if id!=[]:
                new_domain_A.del_atom(id)
        #number 5 here is crystal specific, here is the case for hematite
        for id in self.id_list[:(self.terminated_layer+5)*2]:
            #print id in new_domain_B.id
            new_domain_B.del_atom(id)
        new_domain_A.id=map(lambda x:x+self.domain_tag+'A',new_domain_A.id)
        new_domain_B.id=map(lambda x:x+self.domain_tag+'B',new_domain_B.id)
        return new_domain_A.copy(),new_domain_B.copy()
        
    def _extract_list(self,ref_list,extract_index):
        output_list=[]
        for i in extract_index:
            output_list.append(ref_list[i])
        return output_list
        
    def split_number(self,N_str):
        N_list=[]
        for i in range(len(N_str)):
            N_list.append(int(N_str[i]))
        return N_list
        
    def scale_opt(self,atm_gp_list,scale_factor,sign_values=None,flag='u',ref_v=1.):
        #scale the parameter from first layer atom to deeper layer atom
        #dx,dy,dz,u will decrease inward, oc decrease outward usually
        #and note the ref_v for oc and u is the value for inner most atom, while ref_v for the other parameters are values for outer most atoms
        #atm_gp_list is a list of atom group to consider the scaling operation
        #scale_factor is list of values of scale factor, note accummulated product will be used for scaling
        #flag is the parameter symbol
        #ref_v is the reference value to start off 
        if sign_values==None:
            for i in range(len(atm_gp_list)):
                atm_gp_list[i]._set_func(flag)(ref_v*reduce(mul,scale_factor[:i+1]))
        else:
            for i in range(len(atm_gp_list)):
                atm_gp_list[i]._set_func(flag)(ref_v*sign_values[i]*reduce(mul,scale_factor[:i+1]))
              
    def extract_scale_opts(self,filename):
        self.scale_opt_arr=np.array([[0,0,0,0,0,0]])[0:0]
        f=open(filename)
        lines=f.readlines()
        for line in lines:
            if line[0]!='#':
                line_split=line.rsplit(',')
                self.scale_opt_arr=np.append(self.scale_opt_arr,[[line_split[i] for i in range(6)]],axis=0)
        f.close()
        
    #this function do the same thing as the previous one, but it call open/close file only once by assigning the file content to self.scale_opt_arr
    def scale_opt_batch(self,filename):
        #note:the original name is scale_opt_batch2b
        try:
            test=self.scale_opt_arr
        except:
            self.extract_scale_opts(filename)
        for line in self.scale_opt_arr:
            atm_gp_list=vars(self)[line[0]]
            index_list=self.split_number(line[1])
            scale_factor=vars(self)[line[2]]
            sign_values=0.
            if line[3]=='None':
                sign_values=None
            else:
                sign_values=vars(self)[line[3]]
            flag=line[4]
            ref_v=0.
            try:
                ref_v=float(line[5])
            except:
                ref_v=vars(self)[line[5]]
            
            self.scale_opt(self._extract_list(atm_gp_list,index_list),scale_factor,sign_values,flag,ref_v)
        
    def set_new_vars(self,head_list=['u_Fe_'],N_list=[2]):
    #set new vars 
    #head_list is a list of heading test for a new variable,N_list is the associated number of each set of new variable to be created
        for head,N in zip(head_list,N_list):
            for i in range(N):
                getattr(self.new_var_module,'new_var')(head+str(i+1),1.)
    
    def set_discrete_new_vars_batch(self,filename):
    #set discrete new vars
        f=open(filename)
        lines=f.readlines()
        for line in lines:
            if line[0]!='#':
                line_split=line.rsplit(',')
                #print line_split
                getattr(self.new_var_module,'new_var')(line_split[0],float(line_split[1]))
        f.close()
    
    def norm_sign(self,value,scale=1.):
        if value<=0.5:
            return -scale
        elif value>0.5:
            return scale

    def extract_sim(self,filename):
        self.sim_val=[]
        f=open(filename)
        lines=f.readlines()
        for line in lines:
            if line[0]!='#':
                line_split=list(line.rsplit(','))
                self.sim_val.append(line_split[:-1])
        f.close()
        
    def init_sim_batch(self,filename):
    #note: original name is init_sim_batch2
        try:
            test=self.sim_val
        except:
            self.extract_sim(filename)
        #print self.sim_val
        for line in self.sim_val:

            if (line[0]=='ocu')|(line[0]=='scale'):
                tmp_list=[]
                for i in range(len(line)-2):
                    #print line
                    tmp_list.append(getattr(self.new_var_module,line[i+2]))
                setattr(self,line[1],tmp_list)
            elif line[0]=='ref':
                tmp=getattr(vars(self)[line[2]],line[3])()
                setattr(self,line[1],tmp)
            elif line[0]=='ref_new':
                tmp=getattr(self.new_var_module,line[2])
                setattr(self,line[1],tmp)
            elif line_split[0]=='sign':
                tmp_list=[]
                for i in range(len(line)-2):
                    tmp_list.append(self.norm_sign(getattr(self.new_var_module,line[i+2])))
                setattr(self,line[1],tmp_list)
                    
    def create_grid_number(self,atm,domain):
        atm_coor=extract_coor(domain,atm)
        x,y=atm_coor[0],atm_coor[1]
        #print atm,atm_coor
        a,b,c,d= -0.3,0.3,0.9,1.5
        if ((x>=a) & (x<b))&((y>=a) & (y<b)):
            return 7
        elif ((x>= b) & (x< c))&((y>= a) & (y< b )):
            return 8
        elif ((x>=c) & (x<d))&((y>= a) & (y<b)):
            return 9
        elif ((x>= a) & (x<b))&((y>=b) & (y<c)):
            return 6     
        elif ((x>=b) & (x<c))&((y>=b) & (y<c)):
            return 5 
        elif ((x>=c) & (x<d))&((y>=b) & (y<c)):
            return 4
        elif ((x>= a) & (x<b))&((y>=c) & (y<d)):
            return 1
        elif ((x>=b) & (x<c))&((y>=c) & (y<d)):
            return 2
        elif ((x>=c) & (x<d))&((y>=c) & (y<d)):
            return 3   

    def compare_grid(self,grid1,grid2):
        if grid1==grid2:
            return None
        else:
            return grid_match_lib[grid1][grid2]
            
    def find_neighbors(self,domain,id,searching_range=2.3):
        neighbor_container=[]
        atm_ids=[]
        offset=[]
        full_offset=['+x','-x','+y','-y','+x-y','+x+y','-x+y','-x-y']
        basis=np.array([5.038,5.434,7.3707])
        f1=lambda domain,index:np.array([domain.x[index]+domain.dx1[index]+domain.dx2[index]+domain.dx3[index],domain.y[index]+domain.dy1[index]+domain.dy2[index]+domain.dy3[index],domain.z[index]+domain.dz1[index]+domain.dz2[index]+domain.dz3[index]])*basis
        f2=lambda p1,p2:np.sqrt(np.sum((p1-p2)**2))
        #print domain.id,id
        index=np.where(domain.id==id)[0][0]
        [neighbor_container.append(domain.id[i]) for i in range(len(domain.id)) if (f2(f1(domain,index),f1(domain,i))<=searching_range)&(f2(f1(domain,index),f1(domain,i))!=0.)]
        for i in neighbor_container:
            if i.rsplit('_')[-1] in full_offset:
                atm_ids.append('_'.join(i.rsplit('_')[:-1]))
                offset.append(i.rsplit('_')[-1])
            else:
                atm_ids.append(i)
                offset.append(None)
        return atm_ids,offset
        
    def create_match_lib(self,domain,id_list):
        basis=np.array([5.038,5.434,7.3707])
        match_lib={}
        for i in id_list:
            match_lib[i]=[]
        f1=lambda domain,index:np.array([domain.x[index],domain.y[index],domain.z[index]])*basis
        f2=lambda p1,p2:np.sqrt(np.sum((p1-p2)**2))
        #index=np.where(domain.id==center_atom_id)[0][0]
        for i in range(len(id_list)):
            index_1=np.where(domain.id==id_list[i])[0][0]
            for j in range(len(domain.id)):
                index_2=np.where(domain.id==domain.id[j])[0][0]
                if (f2(f1(domain,index_1),f1(domain,index_2))<2.5):
                    print f2(f1(domain,index_1),f1(domain,index_2))
                    match_lib[id_list[i]].append(domain.id[j])
        return match_lib
               
    def cal_bond_valence1(self,domain,center_atom_id,searching_range=3.,print_file=False):
    #calculate the bond valence for an atom with specified surroundring atoms by using the equation s=exp[(r0-r)/B]
    #the atoms within the searching range (in unit of A) will be counted for the calculation of bond valence
        bond_valence_container={}
        basis=np.array([5.038,5.434,7.3707])
        f1=lambda domain,index:np.array([domain.x[index]+domain.dx1[index],domain.y[index]+domain.dy1[index],domain.z[index]+domain.dz1[index]])*basis
        f2=lambda p1,p2:np.sqrt(np.sum((p1-p2)**2))
        index=np.where(domain.id==center_atom_id)[0][0]
        for i in range(len(domain.id)):
            if (f2(f1(domain,index),f1(domain,i))<=searching_range)&(f2(f1(domain,index),f1(domain,i))!=0.):
                r0=0
                if ((domain.el[index]=='Pb')&(domain.el[i]=='O'))|((domain.el[index]=='O')&(domain.el[i]=='Pb')):r0=2.112
                elif ((domain.el[index]=='Fe')&(domain.el[i]=='O'))|((domain.el[index]=='O')&(domain.el[i]=='Fe')):r0=1.759
                elif ((domain.el[index]=='Sb')&(domain.el[i]=='O'))|((domain.el[index]=='O')&(domain.el[i]=='Sb')):r0=1.973
                else:r0=-10
                bond_valence_container[domain.id[i]]=np.exp((r0-f2(f1(domain,index),f1(domain,i)))/0.37)
        sum_valence=0.
        for key in bond_valence_container.keys():
            sum_valence=sum_valence+bond_valence_container[key]
        bond_valence_container['total_valence']=sum_valence
        if print_file==True:
            f=open('/home/tlab/sphalerite/jackey/model2/files_pb/bond_valence_'+center_atom_id+'.txt','w')
            for i in bond_valence_container.keys():
                s = '%-5s   %7.5e\n' % (i,bond_valence_container[i])
                f.write(s)
            f.close()
        return bond_valence_container
        
    def cal_bond_valence2(self,domain,center_atm,match_list):
        #center_atm='O1',match_list=[['Fe1','Fe2'],['-x','+y']]
        #return a library showing the bond valence contribution to O1
        #calculate the bond valence of (in this case) O1_Fe1, O1_Fe2, where Fe1 and Fe2 have offset defined by '-x' and '+y' respectively.
        #return a lib with keys in match_list[0], the value for each key is the bond valence calculated
        bond_valence_container={}
        match_list.append(0)
        for i in match_list[0]:
            bond_valence_container[i]=0
        basis=np.array([5.038,5.434,7.3707])
        f1=lambda domain,index:np.array([domain.x[index]+domain.dx1[index]+domain.dx2[index]+domain.dx3[index],domain.y[index]+domain.dy1[index]+domain.dy2[index]+domain.dy3[index],domain.z[index]+domain.dz1[index]+domain.dz2[index]+domain.dz3[index]])*basis
        f2=lambda p1,p2:np.sqrt(np.sum((p1-p2)**2))
        
        def _offset_translate(flag):
            if flag=='+x':
                return np.array([1.,0.,0.])*basis
            elif flag=='-x':
                return np.array([-1.,0.,0.])*basis
            elif flag=='+y':
                return np.array([0.,1.,0.])*basis
            elif flag=='-y':
                return np.array([0.,-1.,0.])*basis
            elif flag=='+x+y':
                return np.array([1.,1.,0.])*basis
            elif flag=='+x-y':
                return np.array([1.,-1.,0.])*basis
            elif flag=='-x-y':
                return np.array([-1.,-1.,0.])*basis
            elif flag=='-x+y':
                return np.array([-1.,1.,0.])*basis
            elif flag==None:
                return np.array([0.,0.,0.])*basis
    
        index=np.where(domain.id==center_atm)[0][0]
        for k in range(len(match_list[0])):
            j=match_list[0][k]
            index2=np.where(domain.id==j)[0][0]
            dist=f2(f1(domain,index),f1(domain,index2)+_offset_translate(match_list[1][k]))
            r0=0
            if ((domain.el[index]=='Pb')&(domain.el[index2]=='O'))|((domain.el[index2]=='Pb')&(domain.el[index]=='O')):r0=2.112
            elif ((domain.el[index]=='Fe')&(domain.el[index2]=='O'))|((domain.el[index2]=='Fe')&(domain.el[index]=='O')):r0=1.759
            elif ((domain.el[index]=='Sb')&(domain.el[index2]=='O'))|((domain.el[index2]=='Sb')&(domain.el[index]=='O')):r0=1.973
            else:#when two atoms are too close, the structure explose with high r0, so we are expecting a high bond valence value here.
                if dist<2.:r0=10
                else:r0=0.
                #if (i=='pb1'):
                #print j,str(match_lib[i][1][k]),dist,'pb_coor',f1(domain,index)/basis,'O_coor',(f1(domain,index2)+_offset_translate(match_lib[i][1][k]))/basis,np.exp((r0-dist)/0.37)
            if dist<3.:#take it counted only when they are not two far away
                
                bond_valence_container[j]=np.exp((r0-dist)/0.37)
                #print j,extract_coor(domain,j),dist,bond_valence_container[j]
                match_list[2]=match_list[2]+1
        """
        for i in bond_valence_container.keys():
            #try to add hydrogen or hydrogen bond to the oxygen with 1.6=2*OH, 1.=OH+H, 0.8=OH and 0.2=H
            index=np.where(domain.id==i)[0][0]
            if (domain.el[index]=='O')|(domain.el[index]=='o'):
                case_tag=match_lib[i][2]
                bond_valence_corrected_value=[0.]
                if case_tag==1.:
                    bond_valence_corrected_value=[1.8,1.6,1.2,1.,0.8,0.6,0.4,0.2,0.]
                elif case_tag==2.:
                    bond_valence_corrected_value=[1.6,1.,0.8,0.4,0.2,0.]
                elif case_tag==3.:
                    bond_valence_corrected_value=[0.8,0.2,0.]
                else:pass
                #bond_valence_corrected_value=[1.6,1.,0.8,0.2,0.]
                ref=np.sign(bond_valence_container[i]+np.array(bond_valence_corrected_value)-2.)*(bond_valence_container[i]+np.array(bond_valence_corrected_value)-2.)
                bond_valence_container[i]=bond_valence_container[i]+bond_valence_corrected_value[np.where(ref==np.min(ref))[0][0]]
        """
        cum=sum([bond_valence_container[key] for key in bond_valence_container.keys()])
        bond_valence_container['total']=cum
        return bond_valence_container
        
    def cal_bond_valence3(self,domain,match_lib):
        #match_lib={'O1':[['Fe1','Fe2'],['-x','+y']]}
        #calculate the bond valence of (in this case) O1_Fe1, O1_Fe2, where Fe1 and Fe2 have offset defined by '-x' and '+y' respectively.
        #return a lib with the same key as match_lib, the value for each key is the bond valence calculated
        bond_valence_container={}
        for i in match_lib.keys():
            try:
                match_lib[i][2]=0
            except:
                match_lib[i].append(0)
            bond_valence_container[i]=0
            
        basis=np.array([5.038,5.434,7.3707])
        f1=lambda domain,index:np.array([domain.x[index]+domain.dx1[index]+domain.dx2[index]+domain.dx3[index],domain.y[index]+domain.dy1[index]+domain.dy2[index]+domain.dy3[index],domain.z[index]+domain.dz1[index]+domain.dz2[index]+domain.dz3[index]])*basis
        f2=lambda p1,p2:np.sqrt(np.sum((p1-p2)**2))
        
        def _offset_translate(flag):
            if flag=='+x':
                return np.array([1.,0.,0.])*basis
            elif flag=='-x':
                return np.array([-1.,0.,0.])*basis
            elif flag=='+y':
                return np.array([0.,1.,0.])*basis
            elif flag=='-y':
                return np.array([0.,-1.,0.])*basis
            elif flag=='+x+y':
                return np.array([1.,1.,0.])*basis
            elif flag=='+x-y':
                return np.array([1.,-1.,0.])*basis
            elif flag=='-x-y':
                return np.array([-1.,-1.,0.])*basis
            elif flag=='-x+y':
                return np.array([-1.,1.,0.])*basis
            elif flag==None:
                return np.array([0.,0.,0.])*basis
    
        for i in match_lib.keys():
            index=np.where(domain.id==i)[0][0]
            for k in range(len(match_lib[i][0])):
                j=match_lib[i][0][k]
                index2=np.where(domain.id==j)[0][0]
                dist=f2(f1(domain,index),f1(domain,index2)+_offset_translate(match_lib[i][1][k]))
                r0=0
                if ((domain.el[index]=='Pb')&(domain.el[index2]=='O'))|((domain.el[index2]=='Pb')&(domain.el[index]=='O')):r0=2.112
                elif ((domain.el[index]=='Fe')&(domain.el[index2]=='O'))|((domain.el[index2]=='Fe')&(domain.el[index]=='O')):r0=1.759
                elif ((domain.el[index]=='Sb')&(domain.el[index2]=='O'))|((domain.el[index2]=='Sb')&(domain.el[index]=='O')):r0=1.973
                else:#when two atoms are too close, the structure explose with high r0, so we are expecting a high bond valence value here.
                    if dist<2.:r0=10
                    else:r0=0.
                #if (i=='pb1'):
                    #print j,str(match_lib[i][1][k]),dist,'pb_coor',f1(domain,index)/basis,'O_coor',(f1(domain,index2)+_offset_translate(match_lib[i][1][k]))/basis,np.exp((r0-dist)/0.37)
                if dist<3.:#take it counted only when they are not two far away
                    bond_valence_container[i]=bond_valence_container[i]+np.exp((r0-dist)/0.37)
                    match_lib[i][2]=match_lib[i][2]+1
        
        for i in bond_valence_container.keys():
            #try to add hydrogen or hydrogen bond to the oxygen with 1.6=2*OH, 1.=OH+H, 0.8=OH and 0.2=H
            index=np.where(domain.id==i)[0][0]
            if (domain.el[index]=='O')|(domain.el[index]=='o'):
                case_tag=match_lib[i][2]
                bond_valence_corrected_value=[0.]
                if ((case_tag==1.)&(bond_valence_container[i]<2)):
                    bond_valence_corrected_value=[1.8,1.6,1.2,1.,0.8,0.6,0.4,0.2,0.]
                elif ((case_tag==2.)&(bond_valence_container[i]<2)):
                    bond_valence_corrected_value=[1.6,1.,0.8,0.4,0.2,0.]
                elif ((case_tag==3.)&(bond_valence_container[i]<2)):
                    bond_valence_corrected_value=[0.8,0.2,0.]
                else:pass
                #bond_valence_corrected_value=[1.6,1.,0.8,0.2,0.]
                ref=np.sign(bond_valence_container[i]+np.array(bond_valence_corrected_value)-2.)*(bond_valence_container[i]+np.array(bond_valence_corrected_value)-2.)
                bond_valence_container[i]=bond_valence_container[i]+bond_valence_corrected_value[np.where(ref==np.min(ref))[0][0]]
        
        return bond_valence_container
        
    #set reference coordinate system defined by atoms with ids in domain, create the coordinate transformation matrix between the old and the new ones
    #T is 3by4 matrix with the last column defining the origin of the new coordinate system
    def create_coor_transformation(self,domain,ids):
        origin,p1,p2=extract_coor(domain,ids[0]),extract_coor(domain,ids[1]),extract_coor(domain,ids[2])
        x_v=(p1-origin)/f2(p1,origin)
        p2_o=p2-origin
        z_v=np.cross(x_v,p2_o)
        z_v=z_v/f2(np.array([0.,0.,0.]),z_v)
        y_v=np.cross(z_v,x_v)
        T=f1(x0_v,y0_v,z0_v,x_v,y_v,z_v)
        T=np.append(T,origin[:,np.newaxis],axis=1)
        return T
                
    #extract the r theta and phi for atom with id in the reference coordinate system
    def extract_spherical_pars(self,domain,ref_ids,id):
        T=self.create_coor_transformation(domain,ref_ids)
        coors_old=extract_coor(domain,id)-T[:,-1]
        coors_new=np.dot(T[:,0:-1],coors_old)
        x,y,z=coors_new[0],coors_new[1],coors_new[2]
        r=f2(np.array([0.,0.,0.]),coors_new)
        theta=np.arccos(z/r)
        phi=0
        if (x>0) & (y>0):
            phi=np.arctan(y/x)
        elif (x>0) & (y<0):
            phi=2*np.pi+np.arctan(y/x)
        elif (x<0) & (y>0)|(x<0) & (y<0):
            phi=np.pi+np.arctan(y/x)
        return r,theta,phi       
                            
    #calculate xyz in old coordinate system from spherical system and set it to atom with id    
    def set_sorbate_xyz(self,domain,ref_ids,r_theta_phi,id):
        T=self.create_coor_transformation(domain,ref_ids)
        r,theta,phi=r_theta_phi[0],r_theta_phi[1],r_theta_phi[2]
        x=r*np.sin(theta)*np.cos(phi)
        y=r*np.sin(theta)*np.sin(phi)
        z=r*np.cos(theta)
        coors_new=np.array([x,y,z])
        coors_old=np.dot(inv(T[:,0:-1]),coors_new)+T[:,-1]
        set_coor(domain,id,coors_old)
        
