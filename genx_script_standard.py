#consider two domains (both half layer but different sorbate environment, one with 1pb OH and one with none)
import models.sxrd_test5_sym_new_test_new66_2_2 as model
from models.utils import UserVars
import numpy as np
from operator import mul
from numpy.linalg import inv
from copy import deepcopy

import domain_creator3

################################some functions to be called#######################################
#atoms (sorbates) will be added to position specified by the coor(usually set the coor to the center, then you can easily set dxdy range to [-0.5,0.5] [
def add_atom(domain,ref_coor=[],ids=[],els=[]):
    for i in range(len(ids)):
        domain.add_atom(ids[i],els[i],ref_coor[0],ref_coor[1],ref_coor[2],0.5,1.0,1.0)

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
#file paths
#batch_path_head='/u1/uaf/cqiu/batchfile/'
batch_path_head='D:\\Github\\batchfile\\'
discrete_vars_file_domain1='new_varial_file_standard_A.txt'
sim_batch_file_domain1='sim_batch_file_standard_A.txt'
scale_operation_file_domain1='scale_operation_file_standard_A.txt'
#sorbate ids
sorbate_ids_domain1a=['Pb1_D1A','Pb2_D1A','Pb3_D1A','HO1_D1A','HO2_D1A','HO3_D1A','HO4_D1A','HO5_D1A','HO6_D1A']
sorbate_ids_domain1b=['Pb1_D1B','Pb2_D1B','Pb3_D1B','HO1_D1B','HO2_D1B','HO3_D1B','HO4_D1B','HO5_D1B','HO6_D1B']
sorbate_els_domain1=['Pb','Pb','Pb','O','O','O','O','O','O']
pb_list_domain1a=['Pb1_D1A','Pb2_D1A','Pb3_D1A']
pb_list_domain1b=['Pb1_D1B','Pb2_D1B','Pb3_D1B']
HO_list_domain1a=['HO1_D1A','HO2_D1A','HO3_D1A','HO4_D1A','HO5_D1A','HO6_D1A']
HO_list_domain1b=['HO1_D1B','HO2_D1B','HO3_D1B','HO4_D1B','HO5_D1B','HO6_D1B']
rgh_domain1=UserVars()
#atom ids for grouping(containerB must be the associated chemically equivalent atoms)
ids_domain1A=sorbate_ids_domain1a+["O1_1_0_D1A","O1_2_0_D1A","O1_3_0_D1A","O1_4_0_D1A","Fe1_4_0_D1A","Fe1_6_0_D1A","O1_5_0_D1A","O1_6_0_D1A","O1_7_0_D1A","O1_8_0_D1A","Fe1_8_0_D1A","Fe1_9_0_D1A"]
ids_domain1B=sorbate_ids_domain1b+["O1_7_0_D1B","O1_8_0_D1B","O1_9_0_D1B","O1_10_0_D1B","Fe1_10_0_D1B","Fe1_12_0_D1B","O1_11_0_D1B","O1_12_0_D1B","O1_0_D1B","O2_0_D1B","Fe2_0_D1B","Fe3_0_D1B"]
#group name container(discrete:single atom from each domain, sequence:double atoms at same layer from each domain)
discrete_gp_names=['gp_Pb1_D1','gp_Pb2_D1','gp_Pb3_D1','gp_HO1_D1','gp_HO2_D1','gp_HO3_D1','gp_HO4_D1','gp_HO5_D1','gp_HO6_D1','gp_O1O7_D1','gp_O2O8_D1','gp_O3O9_D1','gp_O4O10_D1','gp_Fe4Fe10_D1','gp_Fe6Fe12_D1','gp_O5O11_D1','gp_O6O12_D1','gp_O7O1_D1','gp_O8O2_D1','gp_Fe8Fe2_D1','gp_Fe9Fe3_D1']
sequence_gp_names=['gp_O1O2_O7O8_D1','gp_Fe2Fe3_Fe8Fe9_D1','gp_O3O4_O9O10_D1','gp_Fe4Fe6_Fe10Fe12_D1','gp_O5O6_O11O12_D1','gp_O7O8_O1O2_D1','gp_Fe8Fe9_Fe2Fe3_D1']
#atom ids being considered for bond valence check
atm_list_1A=['O1_1_0_D1A','O1_2_0_D1A','O1_3_0_D1A','O1_4_0_D1A','O1_5_0_D1A','O1_6_0_D1A','Fe1_4_0_D1A','Fe1_6_0_D1A']
atm_list_1B=['O1_7_0_D1B','O1_8_0_D1B','O1_9_0_D1B','O1_10_0_D1B','O1_11_0_D1B','O1_12_0_D1B','Fe1_10_0_D1B','Fe1_12_0_D1B']
match_order_1A=pb_list_domain1a+HO_list_domain1a+atm_list_1A
match_order_1B=pb_list_domain1b+HO_list_domain1b+atm_list_1B
#id list according to the order in the reference domain   
ref_id_list=["O1_1_0","O1_2_0","Fe1_2_0","Fe1_3_0","O1_3_0","O1_4_0","Fe1_4_0","Fe1_6_0","O1_5_0","O1_6_0","O1_7_0","O1_8_0","Fe1_8_0","Fe1_9_0","O1_9_0","O1_10_0","Fe1_10_0","Fe1_12_0","O1_11_0","O1_12_0",\
"O1_0","O2_0","Fe2_0","Fe3_0","O3_0","O4_0","Fe4_0","Fe6_0","O5_0","O6_0","O7_0","O8_0","Fe8_0","Fe9_0","O9_0","O10_0","Fe10_0","Fe12_0","O11_0","O12_0"]
#the matching row Id information in the symfile
sym_file_Fe=np.array(['Fe1_2_0_D1A','Fe1_3_0_D1A','Fe1_4_0_D1A','Fe1_6_0_D1A','Fe1_8_0_D1A','Fe1_9_0_D1A','Fe1_10_0_D1A','Fe1_12_0_D1A',\
                'Fe2_0_D1A','Fe3_0_D1A','Fe4_0_D1A','Fe6_0_D1A','Fe8_0_D1A','Fe9_0_D1A','Fe10_0_D1A','Fe12_0_D1A'])
sym_file_Fe=np.append(sym_file_Fe,map(lambda x:x[:-1]+'B',sym_file_Fe[4:]))
sym_file_O=np.array(['O1_1_0_D1A','O1_2_0_D1A','O1_3_0_D1A','O1_4_0_D1A','O1_5_0_D1A','O1_6_0_D1A','O1_7_0_D1A','O1_8_0_D1A','O1_9_0_D1A','O1_10_0_D1A','O1_11_0_D1A','O1_12_0_D1A',\
                'O1_0_D1A','O2_0_D1A','O3_0_D1A','O4_0_D1A','O5_0_D1A','O6_0_D1A','O7_0_D1A','O8_0_D1A','O9_0_D1A','O10_0_D1A','O11_0_D1A','O12_0_D1A'])
sym_file_O=np.append(sym_file_O,map(lambda x:x[:-1]+'B',sym_file_O[6:]))
sym_file_HO=np.array(HO_list_domain1a+HO_list_domain1b)
sym_file_O=np.append(sym_file_HO,sym_file_O)
sym_file_Pb=np.array(pb_list_domain1a+pb_list_domain1b)
#symmetry library and id_match in symmetry files 
sym_file={'Fe':batch_path_head+'Fe output file for Genx reading.txt',\
          'O':batch_path_head+'O output file for Genx reading.txt',\
          'Pb':batch_path_head+'Pb output file for Genx reading.txt'}
id_match_in_sym={'Fe':sym_file_Fe,'O':sym_file_O,'Pb':sym_file_Pb}
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
add_atom_in_slab(bulk,batch_path_head+'bulk.str')
add_atom_in_slab(domain0,batch_path_head+'half_layer.str')
add_atom_in_slab(domain0_1,batch_path_head+'full_layer.str')
###################create domain classes and initiate the chemical equivalent domains####################
#when change or create a new domain, make sure the terminated_layer (start from 0)set right
######################################setup domain1############################################
domain_class_1=domain_creator3.domain_creator(ref_domain=domain0,id_list=ref_id_list,terminated_layer=0,new_var_module=rgh_domain1)
domain1A=domain_class_1.domain_A
domain1B=domain_class_1.domain_B
domain_class_1.domain1A=domain1A
domain_class_1.domain1B=domain1B
#Adding sorbates to domain1A and domain1B
add_atom(domain=domain1A,ref_coor=[0.5,0.56955,2.0],ids=sorbate_ids_domain1a,els=sorbate_els_domain1)
add_atom(domain=domain1B,ref_coor=[0.5,0.5,1.5],ids=sorbate_ids_domain1b,els=sorbate_els_domain1)
#set variables
domain_class_1.set_new_vars(head_list=['u_o_n','u_Fe_n','oc_n'],N_list=[4,3,7])
domain_class_1.set_discrete_new_vars_batch(batch_path_head+discrete_vars_file_domain1)
######################################do grouping###############################################
#note the grouping here is on a layer basis, ie atoms of same layer are groupped together
#you may group in symmetry, then atoms of same layer are not independent.
atm_gp_list_domain1=domain_class_1.grouping_sequence_layer(domain=[domain1A,domain1B], first_atom_id=['O1_1_0_D1A','O1_7_0_D1B'],\
                            sym_file=sym_file, id_match_in_sym=id_match_in_sym,layers_N=7,use_sym=False)
domain_class_1.atm_gp_list_domain1=atm_gp_list_domain1
for i in range(len(sequence_gp_names)):vars()[sequence_gp_names[i]]=atm_gp_list_domain1[i]
#you may also only want to group each chemically equivalent atom from two domains
atm_gp_discrete_list_domain1=[]
for i in range(len(ids_domain1A)):
    atm_gp_discrete_list_domain1.append(domain_class_1.grouping_discrete_layer(domain=[domain1A,domain1B],atom_ids=[ids_domain1A[i],ids_domain1B[i]],\
                                                                                        sym_file=sym_file,id_match_in_sym=id_match_in_sym,use_sym=True))
domain_class_1.atm_gp_discrete_list_domain1=atm_gp_discrete_list_domain1    
for i in range(len(discrete_gp_names)):vars()[discrete_gp_names[i]]=atm_gp_discrete_list_domain1[i] 
#print gp_O1O8.ids
#####################################do bond valence matching###################################
match_lib_1A=create_match_lib_before_fitting(domain_class=domain_class_1,domain=domain_class_1.build_super_cell(ref_domain=domain_class_1.create_equivalent_domains_2()[0],rem_atom_ids=['Fe1_2_0_D1A','Fe1_3_0_D1A']),atm_list=atm_list_1A,search_range=2.3)
match_lib_1B=create_match_lib_before_fitting(domain_class=domain_class_1,domain=domain_class_1.build_super_cell(ref_domain=domain_class_1.create_equivalent_domains_2()[1],rem_atom_ids=['Fe1_8_0_D1B','Fe1_9_0_D1B']),atm_list=atm_list_1B,search_range=2.3)
###########################setup domain2################################
#same reference domain,but set the occupancy of second iron layer to 0
#edit and uncomment following segment if you want to consider full layer case


def Sim(data):
    #extract the fitting par values in the associated attribute and then do the scaling(initiation+processing, actually update the fitting parameter values)
    domain_class_1.init_sim_batch2(batch_path_head+sim_batch_file_domain1)
    domain_class_1.scale_opt_batch2b(batch_path_head+scale_operation_file_domain1)
    #create matching lib dynamically during fitting
    match_lib_fitting_1A,match_lib_fitting_1B=deepcopy(match_lib_1A),deepcopy(match_lib_1B)
    create_match_lib_during_fitting(domain_class=domain_class_1,domain=domain1A,atm_list=atm_list_1A,pb_list=pb_list_domain1a,HO_list=HO_list_domain1a,match_lib=match_lib_fitting_1A)
    create_match_lib_during_fitting(domain_class=domain_class_1,domain=domain1B,atm_list=atm_list_1B,pb_list=pb_list_domain1b,HO_list=HO_list_domain1b,match_lib=match_lib_fitting_1B)
    
    F =[]
    beta=rgh.beta
    domain={'domain1A':{'slab':domain1A,'wt':0.5},'domain1B':{'slab':domain1B,'wt':0.5}}
    sample = model.Sample(inst, bulk, domain, unitcell,coherence=False,surface_parms={'delta1':0.,'delta2':0.1391})
    #print domain1A.dx1[0]
    for data_set in data:
        f=np.array([])
        #for extra data set calculate the bond valence instead of structure factor
        if (data_set.extra_data['h'][0]==10):
            bond_valence=domain_class_1.cal_bond_valence3(domain=domain1A,match_lib=match_lib_fitting_1A)
            t=[]
            for i in match_order_1A:
                t.append(bond_valence[i])
            f=np.array(t)
        elif (data_set.extra_data['h'][0]==11):
            bond_valence=domain_class_1.cal_bond_valence3(domain=domain1B,match_lib=match_lib_fitting_1B)
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