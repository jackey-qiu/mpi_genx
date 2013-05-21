#consider two domains (both half layer but different sorbate environment, one with 1pb OH and one with none)
import models.sxrd_test5_sym_new_test_new66_2_2 as model
from models.utils import UserVars
import numpy as np
from operator import mul
from numpy.linalg import inv
from copy import deepcopy
import domain_creator
from domain_creator import add_atom,print_data,create_list,add_atom_in_slab,\
                           create_match_lib_before_fitting,create_match_lib_during_fitting,create_sorbate_ids
'''
  the forms of ids and group names 
###########atm ids################
sorbate: Pb1_D1A, Pb2_D2B, HO1_D1A
water: Os1_D1A
surf atms: O1_1_0_D1A (half layer), O1_1_1_D1A(full layer atms)
############group names###########
gp_Pb1_D1(discrete grouping for sorbate, group u dx dy dz)
gp_O1O7_D1(discrete grouping for surface atms, group dx dy in symmetry)
gp_sorbates_set1_D1(discrete grouping for each set of sorbates, group oc)
gp_waters_set1_D1(discrete grouping for each set of water at same layer, group u, oc and dz)
gp_O1O2_O7O8_D1(sequence grouping for u, oc and dz)
'''
###############################################global vars##################################################
#file paths
batch_path_head='/u1/uaf/cqiu/batchfile/'
#batch_path_head='D:\\Github\\batchfile\\'
Pb_NUMBER=[1]#domain1 has 1 pb and domain2 has 1 pb too
Pb_ATTACH_ATOM=[[['O1_1_0','O1_2_0']]]#The initial lead postion (first one) for domain1 is [0.5,0.56955,2.0]
Pb_ATTACH_ATOM_OFFSET=[[[None,None]]]
O_NUMBER=[[1]]#[[1,2]]means domain has two pb sorbate with one corresponding to monodentate the other one to bidentate
TOP_ANGLE=[[1.0]]#open angel for the surface complex structure
PHI=[[0.]]#rotation angle
WATER_NUMBER=[2]#must be even number considering 2 atoms each layer
V_SHIFT=[3]#vertical shiftment of water molecules, in unit of angstrom
R=[3]#distance bw two waters at each layer in unit of angstrom
ALPHA=[0]#alpha angle used to cal the pos of water molecule
DOMAIN=[1]#1 for half layer and 2 for full layer
DOMAIN_NUMBER=len(DOMAIN)
USE_BV=False
MIRROR=True#consider mirror when adding sorbates?
#sorbate ids
for i in range(DOMAIN_NUMBER):
    #text files
    vars()['discrete_vars_file_domain'+str(int(i+1))]='new_varial_file_domain'+str(int(i+1))+'.txt'
    vars()['sim_batch_file_domain'+str(int(i+1))]='sim_batch_file_domain'+str(int(i+1))+'.txt'
    vars()['scale_operation_file_domain'+str(int(i+1))]='scale_operation_file_domain'+str(int(i+1))+'.txt'
    #sorbate list (HO is oxygen binded to pb and Os is water molecule)
    vars()['pb_list_domain'+str(int(i+1))+'a']=create_sorbate_ids(el='Pb',N=Pb_NUMBER[i],tag='_D'+str(int(i+1))+'A')
    vars()['pb_list_domain'+str(int(i+1))+'b']=create_sorbate_ids(el='Pb',N=Pb_NUMBER[i],tag='_D'+str(int(i+1))+'B')
    vars()['HO_list_domain'+str(int(i+1))+'a']=create_sorbate_ids(el='HO',N=sum(O_NUMBER[i]),tag='_D'+str(int(i+1))+'A')
    vars()['HO_list_domain'+str(int(i+1))+'b']=create_sorbate_ids(el='HO',N=sum(O_NUMBER[i]),tag='_D'+str(int(i+1))+'B') 
    if WATER_NUMBER[i]!=0:
        vars()['Os_list_domain'+str(int(i+1))+'a']=create_sorbate_ids(el='Os',N=WATER_NUMBER[i],tag='_D'+str(int(i+1))+'A')
        vars()['Os_list_domain'+str(int(i+1))+'b']=create_sorbate_ids(el='Os',N=WATER_NUMBER[i],tag='_D'+str(int(i+1))+'B')     
        vars()['sorbate_ids_domain'+str(int(i+1))+'a']=vars()['pb_list_domain'+str(int(i+1))+'a']+vars()['HO_list_domain'+str(int(i+1))+'a']+vars()['Os_list_domain'+str(int(i+1))+'a']
        vars()['sorbate_ids_domain'+str(int(i+1))+'b']=vars()['pb_list_domain'+str(int(i+1))+'b']+vars()['HO_list_domain'+str(int(i+1))+'b']+vars()['Os_list_domain'+str(int(i+1))+'b']
        vars()['sorbate_els_domain'+str(int(i+1))]=['Pb']*Pb_NUMBER[i]+['O']*(sum(O_NUMBER[i])+WATER_NUMBER[i])
    else:
        vars()['sorbate_ids_domain'+str(int(i+1))+'a']=vars()['pb_list_domain'+str(int(i+1))+'a']+vars()['HO_list_domain'+str(int(i+1))+'a']
        vars()['sorbate_ids_domain'+str(int(i+1))+'b']=vars()['pb_list_domain'+str(int(i+1))+'b']+vars()['HO_list_domain'+str(int(i+1))+'b']
        vars()['sorbate_els_domain'+str(int(i+1))]=['Pb']*Pb_NUMBER[i]+['O']*(sum(O_NUMBER[i]))

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
    vars()['match_order_'+str(int(i+1))+'A']=vars()['Os_list_domain'+str(int(i+1))+'a']+vars()['pb_list_domain'+str(int(i+1))+'a']+vars()['HO_list_domain'+str(int(i+1))+'a']+\
                                             map(lambda x:x+'_D'+str(int(i+1))+'A',vars()['atm_list_A_'+str(int(DOMAIN[i]))])
    vars()['match_order_'+str(int(i+1))+'B']=vars()['Os_list_domain'+str(int(i+1))+'b']+vars()['pb_list_domain'+str(int(i+1))+'b']+vars()['HO_list_domain'+str(int(i+1))+'b']+\
                                             map(lambda x:x+'_D'+str(int(i+1))+'B',vars()['atm_list_B_'+str(int(DOMAIN[i]))])
    #the matching row Id information in the symfile
    sym_file_Fe=np.array(['Fe1_2_0','Fe1_3_0','Fe1_4_0','Fe1_6_0','Fe1_8_0','Fe1_9_0','Fe1_10_0','Fe1_12_0',\
                'Fe1_2_1','Fe1_3_1','Fe1_4_1','Fe1_6_1','Fe1_8_1','Fe1_9_1','Fe1_10_1','Fe1_12_1'])
    sym_file_O_1=np.array(['O1_1_0','O1_2_0','O1_3_0','O1_4_0','O1_5_0','O1_6_0','O1_7_0','O1_8_0','O1_9_0','O1_10_0','O1_11_0','O1_12_0',\
                'O1_1_1','O1_2_1','O1_3_1','O1_4_1','O1_5_1','O1_6_1','O1_7_1','O1_8_1','O1_9_1','O1_10_1','O1_11_1','O1_12_1'])
    sym_file_O_2=np.array(['O1_11_t','O1_12_t','O1_1_0','O1_2_0','O1_3_0','O1_4_0','O1_5_0','O1_6_0','O1_7_0','O1_8_0','O1_9_0','O1_10_0','O1_11_0','O1_12_0',\
                'O1_1_1','O1_2_1','O1_3_1','O1_4_1','O1_5_1','O1_6_1','O1_7_1','O1_8_1','O1_9_1','O1_10_1','O1_11_1','O1_12_1'])
    
    vars()['sym_file_Fe_domain'+str(int(i+1))]=np.append(map(lambda x:x+'_D'+str(int(i+1))+'A',sym_file_Fe), map(lambda x:x+'_D'+str(int(i+1))+'B',sym_file_Fe[4:]))          
    vars()['sym_file_O_domain'+str(int(i+1))]=np.append(map(lambda x:x+'_D'+str(int(i+1))+'A',vars()['sym_file_O_'+str(int(DOMAIN[i]))]), map(lambda x:x+'_D'+str(int(i+1))+'B',vars()['sym_file_O_'+str(int(DOMAIN[i]))][6:]))          
    vars()['sym_file_O_domain'+str(int(i+1))]=np.append(vars()['HO_list_domain'+str(int(i+1))+'a']+vars()['Os_list_domain'+str(int(i+1))+'a']+vars()['HO_list_domain'+str(int(i+1))+'b']+vars()['Os_list_domain'+str(int(i+1))+'b'],vars()['sym_file_O_domain'+str(int(i+1))])
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
              'O':batch_path_head+'O output file for Genx reading'+str(sum(O_NUMBER[i])+WATER_NUMBER[i])+'_'+str(int(DOMAIN[i]))+'.txt',\
              'Pb':batch_path_head+'Pb output file for Genx reading'+str(Pb_NUMBER[i])+'_'+str(int(DOMAIN[i]))+'.txt'}
###################create domain classes and initiate the chemical equivalent domains####################
#when change or create a new domain, make sure the terminated_layer (start from 0)set right
######################################setup domains############################################
for i in range(DOMAIN_NUMBER):
    vars()['domain_class_'+str(int(i+1))]=domain_creator.domain_creator(ref_domain=vars()['ref_domain'+str(int(DOMAIN[i]))],id_list=vars()['ref_id_list_'+str(int(DOMAIN[i]))],terminated_layer=0,domain_tag='_D'+str(int(i+1)),new_var_module=vars()['rgh_domain'+str(int(i+1))])
    vars()['domain'+str(int(i+1))+'A']=vars()['domain_class_'+str(int(i+1))].domain_A
    vars()['domain'+str(int(i+1))+'B']=vars()['domain_class_'+str(int(i+1))].domain_B
    vars(vars()['domain_class_'+str(int(i+1))])['domainA']=vars()['domain'+str(int(i+1))+'A']
    vars(vars()['domain_class_'+str(int(i+1))])['domainB']=vars()['domain'+str(int(i+1))+'B']
    #Adding sorbates to domainA and domainB
    pb_coors_a=[]
    O_coors_a=[]
    H2O_coors_a=[]
    for j in range(Pb_NUMBER[i]):
        if len(Pb_ATTACH_ATOM[i][j])==1:
            ids=[Pb_ATTACH_ATOM[i][j][0]+'_D'+str(int(i+1))+'A']
            offset=Pb_ATTACH_ATOM_OFFSET[i][j]
            pb_id=vars()['pb_list_domain'+str(int(i+1))+'a'][j]
            O_indx=[0]+[sum(O_NUMBER[0:ii+1]) for ii in range(len(O_NUMBER[i]))]
            #for [1,2,2], which means inside one domain there are 1OH corresponding to pb1, 2 OH's corresponding to pb2 and so son.
            #will return [0,1,3,5], O_id extract OH according to O_index
            O_id=vars()['HO_list_domain'+str(int(i+1))+'a'][O_index[j]:O_index[j+1]]
            sorbate_coors=vars()['domain_class_'+str(int(i+1))].adding_sorbate_pyramid_monodentate(domain=vars()['domain'+str(int(i+1))+'A'],top_angle=1.,phi=0.,r=2.25,attach_atm_ids=ids,offset=offset,pb_id=pb_id,O_id=O_id,mirror=MIRROR)
            pb_coors_a.append(sorbate_coors[0])
            [O_coors_a.append(sorbate_coors[k]) for k in range(len(sorbate_coors))[1:]]
            #grouping sorbates (each set of Pb and HO, set the occupancy equivalent during fitting)
            pb_id_B=vars()['pb_list_domain'+str(int(i+1))+'b'][j]
            O_id_B=vars()['HO_list_domain'+str(int(i+1))+'b'][O_index[j]:O_index[j+1]]
            sorbate_set_ids=pb_id+O_id+pb_id_B+O_id_B
            N=len(sorbate_set_ids)/2
            vars()['gp_sorbates_set'+str(j+1)+'_D'+str(int(i+1))]=vars()['domain_class_'+str(int(i+1))].grouping_discrete_layer(domain=[vars()['domain'+str(int(i+1))+'A']]*N+[vars()['domain'+str(int(i+1))+'B']]*N,atom_ids=sorbate_set_ids)
        elif len(Pb_ATTACH_ATOM[i][j])==2:
            ids=[Pb_ATTACH_ATOM[i][j][0]+'_D'+str(int(i+1))+'A',Pb_ATTACH_ATOM[i][j][1]+'_D'+str(int(i+1))+'A']
            offset=Pb_ATTACH_ATOM_OFFSET[i][j]
            pb_id=vars()['pb_list_domain'+str(int(i+1))+'a'][j]
            O_index=[0]+[sum(O_NUMBER[i][0:ii+1]) for ii in range(len(O_NUMBER[i]))]
            O_id=vars()['HO_list_domain'+str(int(i+1))+'a'][O_index[j]:O_index[j+1]]
            sorbate_coors=vars()['domain_class_'+str(int(i+1))].adding_sorbate_pyramid_distortion(domain=vars()['domain'+str(int(i+1))+'A'],top_angle=TOP_ANGLE[i][j],phi=PHI[i][j],edge_offset=[0,0],attach_atm_ids=ids,offset=offset,pb_id=pb_id,O_id=O_id,mirror=MIRROR)
            pb_coors_a.append(sorbate_coors[0])
            [O_coors_a.append(sorbate_coors[k]) for k in range(len(sorbate_coors))[1:]]
            #grouping sorbates (each set of Pb and HO, set the occupancy equivalent during fitting, looks like gp_sorbates_set1_D1)
            pb_id_B=vars()['pb_list_domain'+str(int(i+1))+'b'][j]
            O_id_B=vars()['HO_list_domain'+str(int(i+1))+'b'][O_index[j]:O_index[j+1]]
            sorbate_set_ids=pb_id+O_id+pb_id_B+O_id_B
            N=len(sorbate_set_ids)/2
            vars()['gp_sorbates_set'+str(j+1)+'_D'+str(int(i+1))]=vars()['domain_class_'+str(int(i+1))].grouping_discrete_layer(domain=[vars()['domain'+str(int(i+1))+'A']]*N+[vars()['domain'+str(int(i+1))+'B']]*N,atom_ids=sorbate_set_ids)
    #here only consider the sorbates (pb and OH) but not including Os
    sorbate_ids=vars()['pb_list_domain'+str(int(i+1))+'b']+vars()['HO_list_domain'+str(int(i+1))+'b']
    sorbate_els=['Pb']*Pb_NUMBER[i]+['O']*(sum(O_NUMBER[i]))
    add_atom(domain=vars()['domain'+str(int(i+1))+'B'],ref_coor=np.array(pb_coors_a+O_coors_a)*[-1,1,1]-[-1.,0.06955,0.5],ids=sorbate_ids,els=sorbate_els)
    if WATER_NUMBER[i]!=0:#add water molecules if any
        for jj in range(WATER_NUMBER[i]/2):
            O_ids_a=vars()['Os_list_domain'+str(int(i+1))+'a'][jj*2:jj*2+2]
            O_ids_b=vars()['Os_list_domain'+str(int(i+1))+'b'][jj*2:jj*2+2]
            #set the first pb atm to be the ref atm
            H2O_coors_a=vars()['domain_class_'+str(int(i+1))].add_oxygen_pair2(domain=vars()['domain'+str(int(i+1))+'A'],O_ids=O_ids_a,ref_id=vars()['pb_list_domain'+str(int(i+1))+'a'][0],v_shift=V_SHIFT[i],r=R[i],alpha=ALPHA[i])
            add_atom(domain=vars()['domain'+str(int(i+1))+'B'],ref_coor=H2O_coors_a*[-1,1,1]-[-1.,0.06955,0.5],ids=O_ids_b,els=['O','O'])
            #group water molecules at each layer (set equivalent the oc and u during fitting)
            M=len(O_ids_a)
            vars()['gp_waters_set'+str(jj+1)+'_D'+str(int(i+1))]=vars()['domain_class_'+str(int(i+1))].grouping_discrete_layer(domain=[vars()['domain'+str(int(i+1))+'A']]*M+[vars()['domain'+str(int(i+1))+'B']]*M,atom_ids=O_ids_a+O_ids_b)

    #set variables
    vars()['domain_class_'+str(int(i+1))].set_new_vars(head_list=['u_o_n','u_Fe_n','oc_n'],N_list=[4,3,7])
    vars()['domain_class_'+str(int(i+1))].set_discrete_new_vars_batch(batch_path_head+vars()['discrete_vars_file_domain'+str(int(i+1))])
    
######################################do grouping###############################################
for i in range(DOMAIN_NUMBER):
    #note the grouping here is on a layer basis, ie atoms of same layer are groupped together
    #you may group in symmetry, then atoms of same layer are not independent.
    if DOMAIN[i]==1:
        vars()['atm_gp_list_domain'+str(int(i+1))]=vars()['domain_class_'+str(int(i+1))].grouping_sequence_layer(domain=[vars()['domain'+str(int(i+1))+'A'],vars()['domain'+str(int(i+1))+'B']], first_atom_id=['O1_1_0_D'+str(int(i+1))+'A','O1_7_0_D'+str(int(i+1))+'B'],\
                                sym_file=vars()['sym_file_domain'+str(int(i+1))], id_match_in_sym=vars()['id_match_in_sym_domain'+str(int(i+1))],layers_N=10,use_sym=False)
    elif DOMAIN[i]==2:
        vars()['atm_gp_list_domain'+str(int(i+1))]=vars()['domain_class_'+str(int(i+1))].grouping_sequence_layer(domain=[vars()['domain'+str(int(i+1))+'A'],vars()['domain'+str(int(i+1))+'B']], first_atom_id=['O1_11_t_D'+str(int(i+1))+'A','O1_5_0_D'+str(int(i+1))+'B'],\
                                sym_file=vars()['sym_file_domain'+str(int(i+1))], id_match_in_sym=vars()['id_match_in_sym_domain'+str(int(i+1))],layers_N=10,use_sym=False)
    
    vars(vars()['domain_class_'+str(int(i+1))])['atm_gp_list_domain'+str(int(i+1))]=vars()['atm_gp_list_domain'+str(int(i+1))]
    #assign name to each group
    for j in range(len(vars()['sequence_gp_names_domain'+str(int(i+1))])):vars()[vars()['sequence_gp_names_domain'+str(int(i+1))][j]]=vars()['atm_gp_list_domain'+str(int(i+1))][j]
    
    #you may also only want to group each chemically equivalent atom from two domains
    vars()['atm_gp_discrete_list_domain'+str(int(i+1))]=[]
    for j in range(len(vars()['ids_domain'+str(int(i+1))+'A'])):
        vars()['atm_gp_discrete_list_domain'+str(int(i+1))].append(vars()['domain_class_'+str(int(i+1))].grouping_discrete_layer(domain=[vars()['domain'+str(int(i+1))+'A'],vars()['domain'+str(int(i+1))+'B']],\
                                                                   atom_ids=[vars()['ids_domain'+str(int(i+1))+'A'][j],vars()['ids_domain'+str(int(i+1))+'B'][j]],sym_file=vars()['sym_file_domain'+str(int(i+1))],id_match_in_sym=vars()['id_match_in_sym_domain'+str(int(i+1))],use_sym=True))
    vars(vars()['domain_class_'+str(int(i+1))])['atm_gp_discrete_list_domain'+str(int(i+1))]=vars()['atm_gp_discrete_list_domain'+str(int(i+1))]
    for j in range(len(vars()['discrete_gp_names_domain'+str(int(i+1))])):vars()[vars()['discrete_gp_names_domain'+str(int(i+1))][j]]=vars()['atm_gp_discrete_list_domain'+str(int(i+1))][j]

#####################################do bond valence matching###################################
if USE_BV:
    for i in range(DOMAIN_NUMBER):
        if DOMAIN[i]==1:
            vars()['match_lib_'+str(int(i+1))+'A']=create_match_lib_before_fitting(domain_class=vars()['domain_class_'+str(int(i+1))],domain=vars()['domain_class_'+str(int(i+1))].build_super_cell(ref_domain=vars()['domain_class_'+str(int(i+1))].create_equivalent_domains_2()[0],rem_atom_ids=['Fe1_2_0_D'+str(int(i+1))+'A','Fe1_3_0_D'+str(int(i+1))+'A']),atm_list=vars()['atm_list_'+str(int(i+1))+'A'],search_range=2.3)
            vars()['match_lib_'+str(int(i+1))+'B']=create_match_lib_before_fitting(domain_class=vars()['domain_class_'+str(int(i+1))],domain=vars()['domain_class_'+str(int(i+1))].build_super_cell(ref_domain=vars()['domain_class_'+str(int(i+1))].create_equivalent_domains_2()[1],rem_atom_ids=['Fe1_8_0_D'+str(int(i+1))+'B','Fe1_9_0_D'+str(int(i+1))+'B']),atm_list=vars()['atm_list_'+str(int(i+1))+'B'],search_range=2.3)
        elif DOMAIN[i]==2:
            vars()['match_lib_'+str(int(i+1))+'A']=create_match_lib_before_fitting(domain_class=vars()['domain_class_'+str(int(i+1))],domain=vars()['domain_class_'+str(int(i+1))].build_super_cell(ref_domain=vars()['domain_class_'+str(int(i+1))].create_equivalent_domains_2()[0],rem_atom_ids=None),atm_list=vars()['atm_list_'+str(int(i+1))+'A'],search_range=2.3)
            vars()['match_lib_'+str(int(i+1))+'B']=create_match_lib_before_fitting(domain_class=vars()['domain_class_'+str(int(i+1))],domain=vars()['domain_class_'+str(int(i+1))].build_super_cell(ref_domain=vars()['domain_class_'+str(int(i+1))].create_equivalent_domains_2()[1],rem_atom_ids=None),atm_list=vars()['atm_list_'+str(int(i+1))+'B'],search_range=2.3)

VARS=vars()

#print domain_creator3.extract_coor(domain1A,'O1_1_0_D1A')
def Sim(data,VARS=VARS):
    VARS=VARS
    F =[]
    beta=rgh.beta
    total_wt=0
    domain={}

    for i in range(DOMAIN_NUMBER):
        #extract the fitting par values in the associated attribute and then do the scaling(initiation+processing, actually update the fitting parameter values)
        VARS['domain_class_'+str(int(i+1))].init_sim_batch(batch_path_head+VARS['sim_batch_file_domain'+str(int(i+1))])
        VARS['domain_class_'+str(int(i+1))].scale_opt_batch(batch_path_head+VARS['scale_operation_file_domain'+str(int(i+1))])
        #create matching lib dynamically during fitting
        if USE_BV:
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

    #print items
    #print domain_creator3.extract_coor(domain1B,'Pb1_D1B')
    #print_data(N_sorbate=2,N_atm=40,domain=domain1A,z_shift=1,save_file='D://model.xyz')
    if USE_BV:
        bond_valence=[domain_class_1.cal_bond_valence3(domain=VARS['domain'+str(i+1)+'A'],match_lib=vars()['match_lib_fitting_'+str(i+1)+'A']) for i in range(DOMAIN_NUMBER)]
        for bv in bond_valence:
            for key in bv.keys():
                print "%s\t%5.3\n"%(key,bv[key])
    
    for data_set in data:
        f=np.array([])   
        h = data_set.extra_data['h']
        k = data_set.extra_data['k']
        l = data_set.x
        LB = data_set.extra_data['LB']
        dL = data_set.extra_data['dL']
        rough = (1-beta)/((1-beta)**2 + 4*beta*np.sin(np.pi*(l-LB)/dL)**2)**0.5
        f = rough*sample.calc_f(h, k, l)
        F.append(abs(f))
    return F