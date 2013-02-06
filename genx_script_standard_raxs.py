#consider two domains (both half layer but different sorbate environment, one with 1pb OH and one with none)
import models.raxs as model
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
        domain.add_atom(ids[i],els[i],ref_coor[i][0],ref_coor[i][1],ref_coor[i][2],0.5,1.0,1.0)

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

def create_sorbate_ids(el='Pb',N=2,tag='_D1A'):
    id_list=[]
    [id_list.append(el+str(i+1)+tag) for i in range(N)]
    return id_list
###############################################global vars##################################################
#file paths
batch_path_head='/u1/uaf/cqiu/batchfile/'
#batch_path_head='D:\\Github\\batchfile\\'
Pb_NUMBER=[1,1]#domain1 has 1 pb and domain2 has 1 pb too
Pb_COORS=[[[0.5,0.56955,2.0] for i in range(Pb_NUMBER[0])],[[0.5,0.56955,2.0] for i in range(Pb_NUMBER[1])]]#The initial lead postion (first one) for domain1 is [0.5,0.56955,2.0]
O_NUMBER=[2,2]
O_COORS=[[[0.5,0.56955,2.0] for i in range(O_NUMBER[0])],[[0.5,0.56955,2.0] for i in range(O_NUMBER[1])]]
DOMAIN=[1,2]#1 for half layer and 2 for full layer
DOMAIN_NUMBER=len(DOMAIN)
#sorbate ids
for i in range(DOMAIN_NUMBER):
    #text files
    vars()['discrete_vars_file_domain'+str(int(i+1))]='new_varial_file_domain'+str(int(i+1))+'.txt'
    vars()['sim_batch_file_domain'+str(int(i+1))]='sim_batch_file_domain'+str(int(i+1))+'.txt'
    vars()['scale_operation_file_domain'+str(int(i+1))]='scale_operation_file_domain'+str(int(i+1))+'.txt'
    #sorbate list
    vars()['pb_list_domain'+str(int(i+1))+'a']=create_sorbate_ids(el='Pb',N=Pb_NUMBER[i],tag='_D'+str(int(i+1))+'A')
    vars()['pb_list_domain'+str(int(i+1))+'b']=create_sorbate_ids(el='Pb',N=Pb_NUMBER[i],tag='_D'+str(int(i+1))+'B')
    vars()['HO_list_domain'+str(int(i+1))+'a']=create_sorbate_ids(el='HO',N=O_NUMBER[i],tag='_D'+str(int(i+1))+'A')
    vars()['HO_list_domain'+str(int(i+1))+'b']=create_sorbate_ids(el='HO',N=O_NUMBER[i],tag='_D'+str(int(i+1))+'B')    
    vars()['sorbate_ids_domain'+str(int(i+1))+'a']=vars()['pb_list_domain'+str(int(i+1))+'a']+vars()['HO_list_domain'+str(int(i+1))+'a']
    vars()['sorbate_ids_domain'+str(int(i+1))+'b']=vars()['pb_list_domain'+str(int(i+1))+'b']+vars()['HO_list_domain'+str(int(i+1))+'b']
    vars()['sorbate_els_domain'+str(int(i+1))]=['Pb']*Pb_NUMBER[i]+['O']*O_NUMBER[i]
    #user defined variables
    vars()['rgh_domain'+str(int(i+1))]=UserVars()
    #atom ids for grouping(containerB must be the associated chemically equivalent atoms)
    equivalent_atm_list_A_1=["O1_1_0","O1_2_0","Fe1_2_0","Fe1_3_0","O1_3_0","O1_4_0","Fe1_4_0","Fe1_6_0","O1_5_0","O1_6_0","O1_7_0","O1_8_0","Fe1_8_0","Fe1_9_0"]
    equivalent_atm_list_A_2=["O1_11_t","O1_12_t","O1_1_0","O1_2_0","Fe1_2_0","Fe1_3_0","O1_3_0","O1_4_0","Fe1_4_0","Fe1_6_0","O1_5_0","O1_6_0","O1_7_0","O1_8_0"]
    vars()['ids_domain'+str(int(i+1))+'A']=vars()['sorbate_ids_domain'+str(int(i+1))+'a']+map(lambda x:x+'_D'+str(int(i+1))+'A',vars()['equivalent_atm_list_A_'+str(int(DOMAIN[i]))])
    equivalent_atm_list_B_1=["O1_7_0","O1_8_0","Fe1_8_0","Fe1_9_0","O1_9_0","O1_10_0","Fe1_10_0","Fe1_12_0","O1_11_0","O1_12_0","O1_1_1","O1_2_1","Fe1_2_1","Fe1_3_1"]
    equivalent_atm_list_B_2=["O1_5_0","O1_6_0","O1_7_0","O1_8_0","Fe1_8_0","Fe1_9_0","O1_9_0","O1_10_0","Fe1_10_0","Fe1_12_0","O1_11_0","O1_12_0","O1_1_1","O1_2_1"]
    vars()['ids_domain'+str(int(i+1))+'B']=vars()['sorbate_ids_domain'+str(int(i+1))+'b']+map(lambda x:x+'_D'+str(int(i+1))+'B',vars()['equivalent_atm_list_B_'+str(int(DOMAIN[i]))])
    #group name container(discrete:single atom from each domain, sequence:double atoms at same layer from each domain)
    vars()['discrete_gp_names_domain'+str(int(i+1))]=map(lambda x:'gp_'+x.rsplit('_')[0]+'_D'+str(int(i+1)),vars()['sorbate_ids_domain'+str(int(i+1))+'a'])+\
                                                     map(lambda x:'gp_'+x[0].rsplit('_')[0][:-1]+x[0].rsplit('_')[1]+x[1].rsplit('_')[0][:-1]+x[1].rsplit('_')[1]+'_D'+str(int(i+1)),zip(vars()['equivalent_atm_list_A_'+str(int(DOMAIN[i]))],vars()['equivalent_atm_list_B_'+str(int(DOMAIN[i]))]))
    #consider the top 10 atom layers
    atm_sequence_gp_names_1=['O1O2_O7O8','Fe2Fe3_Fe8Fe9','O3O4_O9O10','Fe4Fe6_Fe10Fe12','O5O6_O11O12','O7O8_O1O2','Fe8Fe9_Fe2Fe3','O9O10_O3O4','Fe10Fe12_Fe4Fe6','O11O12_O5O6']
    atm_sequence_gp_names_2=['O11O12_O5O6','O1O2_O7O8','Fe2Fe3_Fe8Fe9','O3O4_O9O10','Fe4Fe6_Fe10Fe12','O5O6_O11O12','O7O8_O1O2','Fe8Fe9_Fe2Fe3','O9O10_O3O4','Fe10Fe12_Fe4Fe6']
    vars()['sequence_gp_names_domain'+str(int(i+1))]=map(lambda x:'gp_'+x+'_D'+str(int(i+1)),vars()['atm_sequence_gp_names_'+str(int(DOMAIN[i]))])
    #atom ids being considered for bond valence check
    atm_list_A_1=['O1_1_0','O1_2_0','O1_3_0','O1_4_0','O1_5_0','O1_6_0','Fe1_4_0','Fe1_6_0']
    atm_list_A_2=['O1_1_0','O1_2_0','O1_3_0','O1_4_0','O1_11_t','O1_12_t','Fe1_2_0','Fe1_3_0']
    
    atm_list_B_1=['O1_7_0','O1_8_0','O1_9_0','O1_10_0','O1_11_0','O1_12_0','Fe1_10_0','Fe1_12_0']
    atm_list_B_2=['O1_7_0','O1_8_0','O1_9_0','O1_10_0','O1_5_0','O1_6_0','Fe1_8_0','Fe1_9_0']
    
    vars()['atm_list_'+str(int(i+1))+'A']=map(lambda x:x+'_D'+str(int(i+1))+'A',vars()['atm_list_A_'+str(int(DOMAIN[i]))])
    vars()['atm_list_'+str(int(i+1))+'B']=map(lambda x:x+'_D'+str(int(i+1))+'B',vars()['atm_list_B_'+str(int(DOMAIN[i]))])
    vars()['match_order_'+str(int(i+1))+'A']=vars()['pb_list_domain'+str(int(i+1))+'a']+vars()['HO_list_domain'+str(int(i+1))+'a']+\
                                             map(lambda x:x+'_D'+str(int(i+1))+'A',vars()['atm_list_A_'+str(int(DOMAIN[i]))])
    vars()['match_order_'+str(int(i+1))+'B']=vars()['pb_list_domain'+str(int(i+1))+'b']+vars()['HO_list_domain'+str(int(i+1))+'b']+\
                                             map(lambda x:x+'_D'+str(int(i+1))+'B',vars()['atm_list_B_'+str(int(DOMAIN[i]))])
    #the matching row Id information in the symfile
    sym_file_Fe=np.array(['Fe1_2_0','Fe1_3_0','Fe1_4_0','Fe1_6_0','Fe1_8_0','Fe1_9_0','Fe1_10_0','Fe1_12_0',\
                'Fe1_2_1','Fe1_3_1','Fe1_4_1','Fe1_6_1','Fe1_8_1','Fe1_9_1','Fe1_10_1','Fe1_12_1'])
    sym_file_O_1=np.array(['O1_1_0','O1_2_0','O1_3_0','O1_4_0','O1_5_0','O1_6_0','O1_7_0','O1_8_0','O1_9_0','O1_10_0','O1_11_0','O1_12_0',\
                'O1_1_1','O1_2_1','O1_3_1','O1_4_1','O1_5_1','O1_6_1','O1_7_1','O1_8_1','O1_9_1','O1_10_1','O1_11_1','O1_12_1'])
    sym_file_O_2=np.array(['O1_11_t','O1_12_t','O1_1_0','O1_2_0','O1_3_0','O1_4_0','O1_5_0','O1_6_0','O1_7_0','O1_8_0','O1_9_0','O1_10_0','O1_11_0','O1_12_0',\
                'O1_1_1','O1_2_1','O1_3_1','O1_4_1','O1_5_1','O1_6_1','O1_7_1','O1_8_1','O1_9_1','O1_10_1','O1_11_1','O1_12_1'])
    
    vars()['sym_file_Fe_domain'+str(int(i+1))]=np.append(map(lambda x:x+'_D'+str(int(i+1))+'A',sym_file_Fe), map(lambda x:x+'_D'+str(int(i+1))+'B',sym_file_Fe[3:]))          
    vars()['sym_file_O_domain'+str(int(i+1))]=np.append(map(lambda x:x+'_D'+str(int(i+1))+'A',vars()['sym_file_O_'+str(int(DOMAIN[i]))]), map(lambda x:x+'_D'+str(int(i+1))+'B',vars()['sym_file_O_'+str(int(DOMAIN[i]))][5:]))          
    vars()['sym_file_O_domain'+str(int(i+1))]=np.append(vars()['HO_list_domain'+str(int(i+1))+'a']+vars()['HO_list_domain'+str(int(i+1))+'b'],vars()['sym_file_O_domain'+str(int(i+1))])
    vars()['sym_file_Pb_domain'+str(int(i+1))]=np.array(vars()['pb_list_domain'+str(int(i+1))+'a']+vars()['pb_list_domain'+str(int(i+1))+'b'])
    vars()['id_match_in_sym_domain'+str(int(i+1))]={'Fe':vars()['sym_file_Fe_domain'+str(int(i+1))],'O':vars()['sym_file_O_domain'+str(int(i+1))],'Pb':vars()['sym_file_Pb_domain'+str(int(i+1))]}

#id list according to the order in the reference domain   
ref_id_list_1=["O1_1_0","O1_2_0","Fe1_2_0","Fe1_3_0","O1_3_0","O1_4_0","Fe1_4_0","Fe1_6_0","O1_5_0","O1_6_0","O1_7_0","O1_8_0","Fe1_8_0","Fe1_9_0","O1_9_0","O1_10_0","Fe1_10_0","Fe1_12_0","O1_11_0","O1_12_0",\
'O1_1_1','O1_2_1','Fe1_2_1','Fe1_3_1','O1_3_1','O1_4_1','Fe1_4_1','Fe1_6_1','O1_5_1','O1_6_1','O1_7_1','O1_8_1','Fe1_8_1','Fe1_9_1','O1_9_1','O1_10_1','Fe1_10_1','Fe1_12_1','O1_11_1','O1_12_1']
ref_id_list_2=["O1_11_t","O1_12_t","O1_1_0","O1_2_0","Fe1_2_0","Fe1_3_0","O1_3_0","O1_4_0","Fe1_4_0","Fe1_6_0","O1_5_0","O1_6_0","O1_7_0","O1_8_0","Fe1_8_0","Fe1_9_0","O1_9_0","O1_10_0","Fe1_10_0","Fe1_12_0","O1_11_0","O1_12_0",\
'O1_1_1','O1_2_1','Fe1_2_1','Fe1_3_1','O1_3_1','O1_4_1','Fe1_4_1','Fe1_6_1','O1_5_1','O1_6_1','O1_7_1','O1_8_1','Fe1_8_1','Fe1_9_1','O1_9_1','O1_10_1','Fe1_10_1','Fe1_12_1','O1_11_1','O1_12_1']

###############################################setting slabs##################################################################    
unitcell = model.UnitCell(5.038, 5.434, 7.3707, 90, 90, 90)
inst = model.Instrument(wavel = .833, alpha = 2.0)
bulk = model.Slab(T_factor='B')
ref_domain1 =  model.Slab(c = 1.0,T_factor='B')
ref_domain2 =  model.Slab(c = 1.0,T_factor='B')
rgh=UserVars()
rgh.new_var('beta', 0.0)
################################################build up ref domains############################################
#add atoms for bulk and two ref domains (ref_domain1<half layer> and ref_domain2<full layer>)
#In those two reference domains, the atoms are ordered according to first hight (z values), then y values
#it is a super surface structure by stacking the surface slab on bulk slab, the repeat vector was counted 
try:
    add_atom_in_slab(bulk,batch_path_head+'bulk.str')
except:
    batch_path_head='D:\\Github\\batchfile\\'
    add_atom_in_slab(bulk,batch_path_head+'bulk.str')
add_atom_in_slab(ref_domain1,batch_path_head+'half_layer2.str')
add_atom_in_slab(ref_domain2,batch_path_head+'full_layer2.str')
#symmetry library, note the Fe output file is the same, but O and Pb output file is different due to different sorbate configuration
for i in range(DOMAIN_NUMBER): 
    vars()['sym_file_domain'+str(int(i+1))]={'Fe':batch_path_head+'Fe output file for Genx reading.txt',\
              'O':batch_path_head+'O output file for Genx reading'+str(O_NUMBER[i])+'_'+str(int(DOMAIN[i]))+'.txt',\
              'Pb':batch_path_head+'Pb output file for Genx reading'+str(Pb_NUMBER[i])+'_'+str(int(DOMAIN[i]))+'.txt'}
###################create domain classes and initiate the chemical equivalent domains####################
#when change or create a new domain, make sure the terminated_layer (start from 0)set right
######################################setup domains############################################
for i in range(DOMAIN_NUMBER):
    vars()['domain_class_'+str(int(i+1))]=domain_creator3.domain_creator(ref_domain=vars()['ref_domain'+str(int(DOMAIN[i]))],id_list=vars()['ref_id_list_'+str(int(DOMAIN[i]))],terminated_layer=0,domain_tag='_D'+str(int(i+1)),new_var_module=vars()['rgh_domain'+str(int(i+1))])
    vars()['domain'+str(int(i+1))+'A']=vars()['domain_class_'+str(int(i+1))].domain_A
    vars()['domain'+str(int(i+1))+'B']=vars()['domain_class_'+str(int(i+1))].domain_B
    vars(vars()['domain_class_'+str(int(i+1))])['domainA']=vars()['domain'+str(int(i+1))+'A']
    vars(vars()['domain_class_'+str(int(i+1))])['domainB']=vars()['domain'+str(int(i+1))+'B']
    #Adding sorbates to domainA and domainB
    add_atom(domain=vars()['domain'+str(int(i+1))+'A'],ref_coor=Pb_COORS[i]+O_COORS[i],ids=vars()['sorbate_ids_domain'+str(int(i+1))+'a'],els=vars()['sorbate_els_domain'+str(int(i+1))])
    add_atom(domain=vars()['domain'+str(int(i+1))+'B'],ref_coor=np.array(Pb_COORS[i]+O_COORS[i])-[0.,0.06955,0.5],ids=vars()['sorbate_ids_domain'+str(int(i+1))+'b'],els=vars()['sorbate_els_domain'+str(int(i+1))])
    #set variables
    vars()['domain_class_'+str(int(i+1))].set_new_vars(head_list=['u_o_n','u_Fe_n','oc_n'],N_list=[4,3,7])
    vars()['domain_class_'+str(int(i+1))].set_discrete_new_vars_batch(batch_path_head+vars()['discrete_vars_file_domain'+str(int(i+1))])
    
######################################do grouping###############################################
for i in range(DOMAIN_NUMBER):
    #note the grouping here is on a layer basis, ie atoms of same layer are groupped together
    #you may group in symmetry, then atoms of same layer are not independent.
    vars()['atm_gp_list_domain'+str(int(i+1))]=vars()['domain_class_'+str(int(i+1))].grouping_sequence_layer(domain=[vars()['domain'+str(int(i+1))+'A'],vars()['domain'+str(int(i+1))+'B']], first_atom_id=['O1_1_0_D'+str(int(i+1))+'A','O1_7_0_D'+str(int(i+1))+'B'],\
                            sym_file=vars()['sym_file_domain'+str(int(i+1))], id_match_in_sym=vars()['id_match_in_sym_domain'+str(int(i+1))],layers_N=10,use_sym=False)

    vars(vars()['domain_class_'+str(int(i+1))])['atm_gp_list_domain'+str(int(i+1))]=vars()['atm_gp_list_domain'+str(int(i+1))]
    for j in range(len(vars()['sequence_gp_names_domain'+str(int(i+1))])):vars()[vars()['sequence_gp_names_domain'+str(int(i+1))][j]]=vars()['atm_gp_list_domain'+str(int(i+1))][j]
    #you may also only want to group each chemically equivalent atom from two domains
    vars()['atm_gp_discrete_list_domain'+str(int(i+1))]=[]
    for j in range(len(vars()['ids_domain'+str(int(i+1))+'A'])):
        vars()['atm_gp_discrete_list_domain'+str(int(i+1))].append(vars()['domain_class_'+str(int(i+1))].grouping_discrete_layer(domain=[vars()['domain'+str(int(i+1))+'A'],vars()['domain'+str(int(i+1))+'B']],\
                                                                   atom_ids=[vars()['ids_domain'+str(int(i+1))+'A'][j],vars()['ids_domain'+str(int(i+1))+'B'][j]],sym_file=vars()['sym_file_domain'+str(int(i+1))],id_match_in_sym=vars()['id_match_in_sym_domain'+str(int(i+1))],use_sym=True))
    vars(vars()['domain_class_'+str(int(i+1))])['atm_gp_discrete_list_domain'+str(int(i+1))]=vars()['atm_gp_discrete_list_domain'+str(int(i+1))]
    for j in range(len(vars()['discrete_gp_names_domain'+str(int(i+1))])):vars()[vars()['discrete_gp_names_domain'+str(int(i+1))][j]]=vars()['atm_gp_discrete_list_domain'+str(int(i+1))][j]

#####################################specify f1f2 here###################################
res_el='Pb'
f1f2_file='pb_new.f1f2'
f1f2=np.loadtxt(batch_path_head+f1f2_file)

VARS=vars()

def Sim(data,VARS=VARS):
    VARS=VARS
    F =[]
    beta=rgh.beta
    total_wt=0
    domain={}

    for i in range(DOMAIN_NUMBER):
        #extract the fitting par values in the associated attribute and then do the scaling(initiation+processing, actually update the fitting parameter values)
        VARS['domain_class_'+str(int(i+1))].init_sim_batch2(batch_path_head+VARS['sim_batch_file_domain'+str(int(i+1))])
        VARS['domain_class_'+str(int(i+1))].scale_opt_batch2b(batch_path_head+VARS['scale_operation_file_domain'+str(int(i+1))])
        
        #set up wt's
        vars()['wt_domain'+str(int(i+1))]=VARS['rgh_domain'+str(int(i+1))].wt
        total_wt=total_wt+vars()['wt_domain'+str(int(i+1))]
    #set up multiple domains
    for i in range(DOMAIN_NUMBER):
        domain['domain'+str(int(i+1))+'A']={'slab':VARS['domain'+str(int(i+1))+'A'],'wt':0.5*vars()['wt_domain'+str(int(i+1))]/total_wt}
        domain['domain'+str(int(i+1))+'B']={'slab':VARS['domain'+str(int(i+1))+'B'],'wt':0.5*vars()['wt_domain'+str(int(i+1))]/total_wt}
    #set up sample
    sample = model.Sample(inst, bulk, domain, unitcell,coherence=False,surface_parms={'delta1':0.,'delta2':0.1391})

    for data_set in data:
        f=np.array([])
        h = data_set.extra_data['h']
        k = data_set.extra_data['k']
        l = data_set.extra_data['l']
        LB = data_set.extra_data['LB']
        dL = data_set.extra_data['dL']
        rough = (1-beta)/((1-beta)**2 + 4*beta*np.sin(np.pi*(l-LB)/dL)**2)**0.5
        f = rough*sample.calc_f(h, k, l,f1f2,res_el)
        F.append(abs(f))
    return F