#consider two domains (both half layer but different sorbate environment, one with 1pb OH and one with none)
import models.sxrd_test5_sym_new_test_new66_2_2 as model
from models.utils import UserVars
import numpy as np
from operator import mul
from numpy.linalg import inv

import domain_creator3

################################some functions to be called#######################################
#atoms (sorbates) will be added to position specified by the coor(usually set the coor to the center, then you can easily set dxdy range to [-0.5,0.5] [
def add_atom(domain,ref_coor=[],ids=[],els=[]):
    for i in range(len(ids)):
        domain.add_atom(ids[i],els[i],ref_coor[0],ref_coor[1],ref_coor[2],0.5,1.0,1.0)

#function to export refined atoms positions after fitting
def print_data(N_sorbate=4,save_file='0001'):
    data=domain1A._extract_values()
    index_all=range(len(data[0]))
    index=index_all[0:20]+index_all[40:40+N_sorbate]
    f=open(save_file,'w')
    for i in index:
        s = '%-5s   %7.5e   %7.5e   %7.5e\n' % (data[3][i],data[0][i]*5.038,(data[1][i]-0.1391)*5.434,(data[2][i]-1.)*7.3707)
        f.write(s)
    f.close()
 
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

###############################################global vars##################################################
sorbate_ids_domain1a=['Pb1a','HO1a','HO2a','HO3a','HO4a']
sorbate_ids_domain1b=['Pb1b','HO1b','HO2b','HO3b','HO4b']
sorbate_els_domain1=['Pb','O','O','O','O',]
pb_list_domain1a=['Pb1a']
pb_list_domain1b=['Pb1b']
HO_list_domain1a=['HO1a','HO2a','HO3a','HO4a']
HO_list_domain1b=['HO1b','HO2b','HO3b','HO4b']
rgh_domain1=UserVars()
#atom ids for grouping(containerB must be the associated chemically equivalent atoms)
ids_domain1A=['Pb1a','HO1a','HO2a','HO3a','HO4a',"O1_1_0","O1_2_0","O1_3_0","O1_4_0","Fe1_4_0","Fe1_6_0","O1_5_0","O1_6_0"]
ids_domain1B=['Pb1b','HO1b','HO2b','HO3b','HO4b',"O1_8_0","O1_7_0","O1_10_0","O1_9_0","Fe1_12_0","Fe1_10_0","O1_12_0","O1_11_0"]
#group name container
discrete_gp_names=['gp_Pb1','gp_HO1','gp_HO2','gp_HO3','gp_HO4','gp_O1O8','gp_O2O7','gp_O3O10','gp_O4O9','gp_Fe4Fe12','gp_Fe6Fe10','gp_O5O12','gp_O6O11']
#atom ids being considered for bond valence check
atm_list_1A=['O1_1_0','O1_2_0','O1_3_0','O1_4_0','O1_5_0','O1_6_0','Fe1_4_0','Fe1_6_0']
atm_list_1B=['O1_8_0','O1_7_0','O1_10_0','O1_9_0','O1_12_0','O1_11_0','Fe1_12_0','Fe1_10_0']
match_order_1A=pb_list_domain1a+HO_list_domain1a+atm_list_1A
match_order_1B=pb_list_domain1b+HO_list_domain1b+atm_list_1B
#id list according to the order in the reference domain   
ref_id_list=["O1_1_0","O1_2_0","Fe1_2_0","Fe1_3_0","O1_3_0","O1_4_0","Fe1_4_0","Fe1_6_0","O1_5_0","O1_6_0","O1_7_0","O1_8_0","Fe1_8_0","Fe1_9_0","O1_9_0","O1_10_0","Fe1_10_0","Fe1_12_0","O1_11_0","O1_12_0",\
"O_1_0","O_2_0","Fe_2_0","Fe_3_0","O_3_0","O_4_0","Fe_4_0","Fe_6_0","O_5_0","O_6_0","O_7_0","O_8_0","Fe_8_0","Fe_9_0","O_9_0","O_10_0","Fe_10_0","Fe_12_0","O_11_0","O_12_0"]
#the matching row Id information in the symfile
sym_file_Fe=np.array(['Fe1_0','Fe2_0','Fe3_0','Fe4_0','Fe5_0','Fe6_0','Fe7_0','Fe8_0','Fe9_0','Fe10_0','Fe11_0','Fe12_0',\
    'Fe1_1_0','Fe1_2_0','Fe1_3_0','Fe1_4_0','Fe1_5_0','Fe1_6_0','Fe1_7_0','Fe1_8_0','Fe1_9_0','Fe1_10_0','Fe1_11_0','Fe1_12_0'])
sym_file_O=np.array(['O1_0','O2_0','O3_0','O4_0','O5_0','O6_0','O7_0','O8_0','O9_0','O10_0','O11_0','O12_0',\
    'O1_1_0','O1_2_0','O1_3_0','O1_4_0','O1_5_0','O1_6_0','O1_7_0','O1_8_0','O1_9_0','O1_10_0','O1_11_0','O1_12_0'])
#file paths
batch_path_head='C:\\cygwin\\home\\Canrong Qiu\\batchfile\\'
discrete_vars_file_domain1='new_varial_file_standard_A.txt'
sim_batch_file_domain1='sim_batch_file_standard_A.txt'
scale_operation_file_domain1='scale_operation_file_standard_A.txt'
sym_file_head='/home/jackey/genx_data/'
###############################################setting slabs##################################################################    
unitcell = model.UnitCell(5.038, 5.434, 7.3707, 90, 90, 90)
inst = model.Instrument(wavel = .833, alpha = 2.0)
bulk = model.Slab(T_factor='B')
domain0 =  model.Slab(c = 1.0,T_factor='B')
domain0_1 =  model.Slab(c = 1.0,T_factor='B')
rgh=UserVars()
rgh.new_var('beta', 0.0)
################################################build up ref domains############################################
#add atoms for bulk and two ref domains (domain0<half layer> and domain0_1<full layer>)
#In those two reference domains, the atoms are ordered according to first hight (z values), then y values
#it is a super surface structure by stacking the surface slab on bulk slab, the repeat vector was counted 
add_atom_in_slab(bulk,batch_path_head+'atom files in bulk.txt')
add_atom_in_slab(domain0,batch_path_head+'atom files in domain0.txt')
add_atom_in_slab(domain0_1,batch_path_head+'atom files in domain0_1.txt')
###################create domain classes and initiate the chemical equivalent domains####################
#when change or create a new domain, make sure the terminated_layer (start from 0)set right
######################################setup domain1############################################
domain_class_1=domain_creator3.domain_creator(ref_domain=domain0,id_list=ref_id_list,terminated_layer=0,new_var_module=rgh_domain1)
domain1A=domain_class_1.domain_A
domain1B=domain_class_1.domain_B
domain_class_1.domain1A=domain1A
domain_class_1.domain1B=domain1B
#Adding sorbates to domain1A and domain1B
add_atom(domain=domain1A,ref_coor=[0.5,0.6391,2.0],ids=sorbate_ids_domain1a,els=sorbate_els_domain1)
add_atom(domain=domain1B,ref_coor=[0.5,0.6391,1.5],ids=sorbate_ids_domain1b,els=sorbate_els_domain1)
#set variables
domain_class_1.set_new_vars(head_list=['u_o_n','u_Fe_n','oc_n'],N_list=[4,3,7])
domain_class_1.set_discrete_new_vars_batch(batch_path_head+discrete_vars_file_domain1)
######################################do grouping###############################################
#note the grouping here is on a layer basis, ie atoms of same layer are groupped together
#you may group in symmetry, then atoms of same layer are not independent.
atm_gp_list_domain1=domain_class_1.grouping_sequence_layer(domain=[domain1A,domain1B], first_atom_id=['O1_1_0','O1_7_0'],\
    sym_file=None,id_match_in_sym={'Fe':sym_file_Fe,'O':sym_file_O},layers_N=7,use_sym=False)
domain_class_1.atm_gp_list_domain1=atm_gp_list_domain1
#you may also only want to group each chemically equivalent atom from two domains
atm_gp_discrete_list_domain1=[]
for i in range(len(ids_domain1A)):
    atm_gp_discrete_list_domain1.append(domain_class_1.grouping_discrete_layer(domain=[domain1A,domain1B],atom_ids=[ids_domain1A[i],ids_domain1B[i]]))
domain_class_1.atm_gp_discrete_list_domain1=atm_gp_discrete_list_domain1    
for i in range(len(discrete_gp_names)):vars()[discrete_gp_names[i]]=atm_gp_discrete_list_domain1[i] 
#print gp_O1O8.ids
#####################################do bond valence matching###################################
match_lib_1A=create_match_lib_before_fitting(domain_class=domain_class_1,domain=domain_class_1.build_super_cell(ref_domain=domain0,rem_atom_ids=["Fe1_3_0","Fe1_2_0"]),atm_list=atm_list_1A,search_range=2.3)
match_lib_1B=create_match_lib_before_fitting(domain_class=domain_class_1,domain=domain_class_1.build_super_cell(ref_domain=domain0,rem_atom_ids=["O1_1_0","O1_2_0","Fe1_2_0","Fe1_3_0","O1_3_0","O1_4_0","Fe1_4_0","Fe1_6_0","O1_5_0","O1_6_0","Fe1_9_0","Fe1_8_0"]),\
                                                                                                                    atm_list=atm_list_1B,search_range=2.3)
###########################setup domain2################################
#same reference domain,but set the occupancy of second iron layer to 0
#edit and uncomment following segment if you want to consider full layer case


def Sim(data):
    #extract the fitting par values in the associated attribute and then do the scaling(initiation+processing, actually update the fitting parameter values)
    domain_class_1.init_sim_batch2(batch_path_head+sim_batch_file_domain1)
    domain_class_1.scale_opt_batch2b(batch_path_head+scale_operation_file_domain1)
    #create matching lib dynamically during fitting
    create_match_lib_during_fitting(domain_class=domain_class_1,domain=domain1A,atm_list=atm_list_1A,pb_list=pb_list_domain1a,HO_list=HO_list_domain1a,match_lib=match_lib_1A)
    create_match_lib_during_fitting(domain_class=domain_class_1,domain=domain1B,atm_list=atm_list_1B,pb_list=pb_list_domain1b,HO_list=HO_list_domain1b,match_lib=match_lib_1B)
    
    F =[]
    beta=rgh.beta
    domain={'domain1A':{'slab':domain1A,'wt':0.5},'domain1B':{'slab':domain1B,'wt':0.5}}
    sample = model.Sample(inst, bulk, domain, unitcell,coherence=False,surface_parms={'delta1':0.,'delta2':0.1391})
    #print domain1A.dx1[0]
    for data_set in data:
        f=np.array([])
        #for extra data set calculate the bond valence instead of structure factor
        if (data_set.extra_data['h'][0]==10):
            bond_valence=domain_class_1.cal_bond_valence3(domain=domain1A,match_lib=match_lib_1A)
            t=[]
            for i in match_order_1A:
                t.append(bond_valence[i])
            f=np.array(t)
        elif (data_set.extra_data['h'][0]==11):
            bond_valence=domain_class_1.cal_bond_valence3(domain=domain1B,match_lib=match_lib_1B)
            t=[]
            for i in match_order_1B:
                t.append(bond_valence[i])
            f=np.array(t)
        else:
            h = data_set.extra_data['h']
            k = data_set.extra_data['k']
            l = data_set.x
            LB = data_set.extra_data['LB']
            dL = data_set.extra_data['dL']
            rough = (1-beta)/((1-beta)**2 + 4*beta*np.sin(np.pi*(l-LB)/dL)**2)**0.5
            f = rough*sample.calc_f(h, k, l)
        i = abs(f)
        F.append(i)
    return F