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

def create_sorbate_ids(el='Pb',N=2,tag='_D1A'):
    id_list=[]
    [id_list.append(el+str(i+1)+tag) for i in range(N)]
    return id_list
###############################################global vars##################################################
#file paths
#batch_path_head='/u1/uaf/cqiu/batchfile/'
batch_path_head='D:\\Github\\batchfile\\'
Pb_NUMBER=1
O_NUMBER=4
DOMAIN_NUMBER=2
#sorbate ids
for i in range(DOMAIN_NUMBER):
    #text files
    vars()['discrete_vars_file_domain'+str(int(i+1))]='new_varial_file_domain'+str(int(i+1))+'.txt'
    vars()['sim_batch_file_domain'+str(int(i+1))]='sim_batch_file_domain'+str(int(i+1))+'.txt'
    vars()['scale_operation_file_domain'+str(int(i+1))]='scale_operation_file_domain'+str(int(i+1))+'.txt'
    #sorbate list
    vars()['pb_list_domain'+str(int(i+1))+'a']=create_sorbate_ids(el='Pb',N=Pb_NUMBER,tag='_D'+str(int(i+1))+'A')
    vars()['pb_list_domain'+str(int(i+1))+'b']=create_sorbate_ids(el='Pb',N=Pb_NUMBER,tag='_D'+str(int(i+1))+'B')
    vars()['HO_list_domain'+str(int(i+1))+'a']=create_sorbate_ids(el='HO',N=O_NUMBER,tag='_D'+str(int(i+1))+'A')
    vars()['HO_list_domain'+str(int(i+1))+'b']=create_sorbate_ids(el='HO',N=O_NUMBER,tag='_D'+str(int(i+1))+'B')    
    vars()['sorbate_ids_domain'+str(int(i+1))+'a']=vars()['pb_list_domain'+str(int(i+1))+'a']+vars()['HO_list_domain'+str(int(i+1))+'a']
    vars()['sorbate_ids_domain'+str(int(i+1))+'b']=vars()['pb_list_domain'+str(int(i+1))+'b']+vars()['HO_list_domain'+str(int(i+1))+'b']
    vars()['sorbate_els_domain'+str(int(i+1))]=['Pb']*Pb_NUMBER+['O']*O_NUMBER
    #user defined variables
    vars()['rgh_domain'+str(int(i+1))]=UserVars()
    #atom ids for grouping(containerB must be the associated chemically equivalent atoms)
    equivalent_atm_list_A=["O1_1_0","O1_2_0","O1_3_0","O1_4_0","Fe1_4_0","Fe1_6_0","O1_5_0","O1_6_0","O1_7_0","O1_8_0","Fe1_8_0","Fe1_9_0"]
    vars()['ids_domain'+str(int(i+1))+'A']=vars()['sorbate_ids_domain'+str(int(i+1))+'a']+map(lambda x:x+'_D'+str(int(i+1))+'A',equivalent_atm_list_A)
    equivalent_atm_list_B=["O1_7_0","O1_8_0","O1_9_0","O1_10_0","Fe1_10_0","Fe1_12_0","O1_11_0","O1_12_0","O1_1_1","O1_2_1","Fe1_2_1","Fe1_3_1"]
    vars()['ids_domain'+str(int(i+1))+'B']=vars()['sorbate_ids_domain'+str(int(i+1))+'b']+map(lambda x:x+'_D'+str(int(i+1))+'B',equivalent_atm_list_B)
    #group name container(discrete:single atom from each domain, sequence:double atoms at same layer from each domain)
    vars()['discrete_gp_names_domain'+str(int(i+1))]=map(lambda x:'gp_'+x.rsplit('_')[0]+'_D'+str(int(i+1)),vars()['sorbate_ids_domain'+str(int(i+1))+'a'])+\
                                                     map(lambda x:'gp_'+x[0].rsplit('_')[0][:-1]+x[0].rsplit('_')[1]+x[1].rsplit('_')[0][:-1]+x[1].rsplit('_')[1]+'_D'+str(int(i+1)),zip(equivalent_atm_list_A,equivalent_atm_list_B))
    atm_sequence_gp_names=['O1O2_O7O8','Fe2Fe3_Fe8Fe9','O3O4_O9O10','Fe4Fe6_Fe10Fe12','O5O6_O11O12','O7O8_O1O2','Fe8Fe9_Fe2Fe3']
    vars()['sequence_gp_names_domain'+str(int(i+1))]=map(lambda x:'gp_'+x+'_D'+str(int(i+1)),atm_sequence_gp_names)
    #atom ids being considered for bond valence check
    atm_list_A=['O1_1_0','O1_2_0','O1_3_0','O1_4_0','O1_5_0','O1_6_0','Fe1_4_0','Fe1_6_0']
    atm_list_B=['O1_7_0','O1_8_0','O1_9_0','O1_10_0','O1_11_0','O1_12_0','Fe1_10_0','Fe1_12_0']
    vars()['atm_list_'+str(int(i+1))+'A']=map(lambda x:x+'_D'+str(int(i+1))+'A',atm_list_A)
    vars()['atm_list_'+str(int(i+1))+'B']=map(lambda x:x+'_D'+str(int(i+1))+'B',atm_list_B)
    vars()['match_order_'+str(int(i+1))+'A']=vars()['pb_list_domain'+str(int(i+1))+'a']+vars()['HO_list_domain'+str(int(i+1))+'a']+\
                                             map(lambda x:x+'_D'+str(int(i+1))+'A',atm_list_A)
    vars()['match_order_'+str(int(i+1))+'B']=vars()['pb_list_domain'+str(int(i+1))+'b']+vars()['HO_list_domain'+str(int(i+1))+'b']+\
                                             map(lambda x:x+'_D'+str(int(i+1))+'B',atm_list_B)
    #the matching row Id information in the symfile
    sym_file_Fe=np.array(['Fe1_2_0','Fe1_3_0','Fe1_4_0','Fe1_6_0','Fe1_8_0','Fe1_9_0','Fe1_10_0','Fe1_12_0',\
                'Fe1_2_1','Fe1_3_1','Fe1_4_1','Fe1_6_1','Fe1_8_1','Fe1_9_1','Fe1_10_1','Fe1_12_1'])
    sym_file_O=np.array(['O1_1_0','O1_2_0','O1_3_0','O1_4_0','O1_5_0','O1_6_0','O1_7_0','O1_8_0','O1_9_0','O1_10_0','O1_11_0','O1_12_0',\
                'O1_1_1','O1_2_1','O1_3_1','O1_4_1','O1_5_1','O1_6_1','O1_7_1','O1_8_1','O1_9_1','O1_10_1','O1_11_1','O1_12_1'])
    vars()['sym_file_Fe_domain'+str(int(i+1))]=np.append(map(lambda x:x+'_D'+str(int(i+1))+'A',sym_file_Fe), map(lambda x:x+'_D'+str(int(i+1))+'B',sym_file_Fe))          
    vars()['sym_file_O_domain'+str(int(i+1))]=np.append(map(lambda x:x+'_D'+str(int(i+1))+'A',sym_file_O), map(lambda x:x+'_D'+str(int(i+1))+'B',sym_file_O))          
    vars()['sym_file_O_domain'+str(int(i+1))]=np.append(vars()['HO_list_domain'+str(int(i+1))+'a']+vars()['HO_list_domain'+str(int(i+1))+'b'],vars()['sym_file_O_domain'+str(int(i+1))])
    vars()['sym_file_Pb_domain'+str(int(i+1))]=np.array(vars()['pb_list_domain'+str(int(i+1))+'a']+vars()['pb_list_domain'+str(int(i+1))+'b'])
    vars()['id_match_in_sym_domain'+str(int(i+1))]={'Fe':vars()['sym_file_Fe_domain'+str(int(i+1))],'O':vars()['sym_file_O_domain'+str(int(i+1))],'Pb':vars()['sym_file_Pb_domain'+str(int(i+1))]}

#id list according to the order in the reference domain   
ref_id_list=["O1_1_0","O1_2_0","Fe1_2_0","Fe1_3_0","O1_3_0","O1_4_0","Fe1_4_0","Fe1_6_0","O1_5_0","O1_6_0","O1_7_0","O1_8_0","Fe1_8_0","Fe1_9_0","O1_9_0","O1_10_0","Fe1_10_0","Fe1_12_0","O1_11_0","O1_12_0",\
'O1_1_1','O1_2_1','Fe1_2_1','Fe1_3_1','O1_3_1','O1_4_1','Fe1_4_1','Fe1_6_1','O1_5_1','O1_6_1','O1_7_1','O1_8_1','Fe1_8_1','Fe1_9_1','O1_9_1','O1_10_1','Fe1_10_1','Fe1_12_1','O1_11_1','O1_12_1']

#symmetry library 
sym_file={'Fe':batch_path_head+'Fe output file for Genx reading.txt',\
          'O':batch_path_head+'O output file for Genx reading.txt',\
          'Pb':batch_path_head+'Pb output file for Genx reading.txt'}

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
add_atom_in_slab(domain0,batch_path_head+'half_layer2.str')
add_atom_in_slab(domain0_1,batch_path_head+'full_layer2.str')
###################create domain classes and initiate the chemical equivalent domains####################
#when change or create a new domain, make sure the terminated_layer (start from 0)set right
######################################setup domains############################################
for i in range(DOMAIN_NUMBER):
    vars()['domain_class_'+str(int(i+1))]=domain_creator3.domain_creator(ref_domain=domain0,id_list=ref_id_list,terminated_layer=0,domain_tag='_D'+str(int(i+1)),new_var_module=vars()['rgh_domain'+str(int(i+1))])
    vars()['domain'+str(int(i+1))+'A']=vars()['domain_class_'+str(int(i+1))].domain_A
    vars()['domain'+str(int(i+1))+'B']=vars()['domain_class_'+str(int(i+1))].domain_B
    vars(vars()['domain_class_'+str(int(i+1))])['domainA']=vars()['domain'+str(int(i+1))+'A']
    vars(vars()['domain_class_'+str(int(i+1))])['domainB']=vars()['domain'+str(int(i+1))+'B']
    #Adding sorbates to domainA and domainB
    add_atom(domain=vars()['domain'+str(int(i+1))+'A'],ref_coor=[0.5,0.56955,2.0],ids=vars()['sorbate_ids_domain'+str(int(i+1))+'a'],els=vars()['sorbate_els_domain'+str(int(i+1))])
    add_atom(domain=vars()['domain'+str(int(i+1))+'B'],ref_coor=[0.5,0.5,1.5],ids=vars()['sorbate_ids_domain'+str(int(i+1))+'b'],els=vars()['sorbate_els_domain'+str(int(i+1))])
    #set variables
    vars()['domain_class_'+str(int(i+1))].set_new_vars(head_list=['u_o_n','u_Fe_n','oc_n'],N_list=[4,3,7])
    vars()['domain_class_'+str(int(i+1))].set_discrete_new_vars_batch(batch_path_head+vars()['discrete_vars_file_domain'+str(int(i+1))])
    
######################################do grouping###############################################
for i in range(DOMAIN_NUMBER):
    #note the grouping here is on a layer basis, ie atoms of same layer are groupped together
    #you may group in symmetry, then atoms of same layer are not independent.
    vars()['atm_gp_list_domain'+str(int(i+1))]=vars()['domain_class_'+str(int(i+1))].grouping_sequence_layer(domain=[vars()['domain'+str(int(i+1))+'A'],vars()['domain'+str(int(i+1))+'B']], first_atom_id=['O1_1_0_D'+str(int(i+1))+'A','O1_7_0_D'+str(int(i+1))+'B'],\
                            sym_file=sym_file, id_match_in_sym=vars()['id_match_in_sym_domain'+str(int(i+1))],layers_N=7,use_sym=False)

    vars(vars()['domain_class_'+str(int(i+1))])['atm_gp_list_domain'+str(int(i+1))]=vars()['atm_gp_list_domain'+str(int(i+1))]
    for j in range(len(vars()['sequence_gp_names_domain'+str(int(i+1))])):vars()[vars()['sequence_gp_names_domain'+str(int(i+1))][j]]=vars()['atm_gp_list_domain'+str(int(i+1))][j]
    #you may also only want to group each chemically equivalent atom from two domains
    vars()['atm_gp_discrete_list_domain'+str(int(i+1))]=[]
    for j in range(len(vars()['ids_domain'+str(int(i+1))+'A'])):
        vars()['atm_gp_discrete_list_domain'+str(int(i+1))].append(vars()['domain_class_'+str(int(i+1))].grouping_discrete_layer(domain=[vars()['domain'+str(int(i+1))+'A'],vars()['domain'+str(int(i+1))+'B']],\
                                                                   atom_ids=[vars()['ids_domain'+str(int(i+1))+'A'][j],vars()['ids_domain'+str(int(i+1))+'B'][j]],sym_file=sym_file,id_match_in_sym=vars()['id_match_in_sym_domain'+str(int(i+1))],use_sym=True))
    vars(vars()['domain_class_'+str(int(i+1))])['atm_gp_discrete_list_domain'+str(int(i+1))]=vars()['atm_gp_discrete_list_domain'+str(int(i+1))]
    for j in range(len(vars()['discrete_gp_names_domain'+str(int(i+1))])):vars()[vars()['discrete_gp_names_domain'+str(int(i+1))][j]]=vars()['atm_gp_discrete_list_domain'+str(int(i+1))][j]

#####################################do bond valence matching###################################
for i in range(DOMAIN_NUMBER):
    vars()['match_lib_'+str(int(i+1))+'A']=create_match_lib_before_fitting(domain_class=vars()['domain_class_'+str(int(i+1))],domain=vars()['domain_class_'+str(int(i+1))].build_super_cell(ref_domain=vars()['domain_class_'+str(int(i+1))].create_equivalent_domains_2()[0],rem_atom_ids=['Fe1_2_0_D'+str(int(i+1))+'A','Fe1_3_0_D'+str(int(i+1))+'A']),atm_list=vars()['atm_list_'+str(int(i+1))+'A'],search_range=2.3)
    vars()['match_lib_'+str(int(i+1))+'B']=create_match_lib_before_fitting(domain_class=vars()['domain_class_'+str(int(i+1))],domain=vars()['domain_class_'+str(int(i+1))].build_super_cell(ref_domain=vars()['domain_class_'+str(int(i+1))].create_equivalent_domains_2()[1],rem_atom_ids=['Fe1_8_0_D'+str(int(i+1))+'B','Fe1_9_0_D'+str(int(i+1))+'B']),atm_list=vars()['atm_list_'+str(int(i+1))+'B'],search_range=2.3)
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
        #create matching lib dynamically during fitting
        vars()['match_lib_fitting_'+str(i+1)+'A'],vars()['match_lib_fitting_'+str(i+1)+'B']=deepcopy(VARS['match_lib_'+str(i+1)+'A']),deepcopy(VARS['match_lib_'+str(i+1)+'B'])
        create_match_lib_during_fitting(domain_class=VARS['domain_class_'+str(int(i+1))],domain=VARS['domain'+str(int(i+1))+'A'],atm_list=VARS['atm_list_'+str(int(i+1))+'A'],pb_list=VARS['pb_list_domain'+str(int(i+1))+'a'],HO_list=VARS['HO_list_domain'+str(int(i+1))+'a'],match_lib=vars()['match_lib_fitting_'+str(int(i+1))+'A'])
        #create_match_lib_during_fitting(domain_class=VARS['domain_class_'+str(int(i+1))],domain=VARS['domain'+str(int(i+1))+'B'],atm_list=VARS['atm_list_'+str(int(i+1))+'B'],pb_list=VARS['pb_list_domain'+str(int(i+1))+'b'],HO_list=VARS['HO_list_domain'+str(int(i+1))+'b'],match_lib=vars()['match_lib_fitting_'+str(int(i+1))+'B'])
        #set up wt's
        vars()['wt_domain'+str(int(i+1))]=VARS['rgh_domain'+str(int(i+1))].wt
        total_wt=total_wt+vars()['wt_domain'+str(int(i+1))]
    #set up multiple domains
    for i in range(DOMAIN_NUMBER):
        domain['domain'+str(int(i+1))+'A']={'slab':VARS['domain'+str(int(i+1))+'A'],'wt':0.5*vars()['wt_domain'+str(int(i+1))]/total_wt}
        domain['domain'+str(int(i+1))+'B']={'slab':VARS['domain'+str(int(i+1))+'B'],'wt':0.5*vars()['wt_domain'+str(int(i+1))]/total_wt}
    #set up sample
    sample = model.Sample(inst, bulk, domain, unitcell,coherence=False,surface_parms={'delta1':0.,'delta2':0.1391})
    #extra tag for extra dataset consideration(10-->domain1, 11-->domain2, 12-->domain3, and so on, only consider one equivalent domain)
    extra_tag=[10+i for i in range(DOMAIN_NUMBER)]
    for data_set in data:
        f=np.array([])
        #for extra data set calculate the bond valence instead of structure factor
        if (data_set.extra_data['h'][0] in extra_tag):
            tag=int(data_set.extra_data['h'][0]-10+1)
            bond_valence=domain_class_1.cal_bond_valence3(domain=VARS['domain'+str(tag)+'A'],match_lib=vars()['match_lib_fitting_'+str(tag)+'A'])
            t=[]
            for i in VARS['match_order_'+str(tag)+'A']:
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
        F.append(abs(f))
    return F