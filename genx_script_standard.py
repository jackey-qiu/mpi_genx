#consider two domains (both half layer but different sorbate environment, one with 1pb OH and one with none)
import models.sxrd_test5_sym_new_test_new66_2 as model
from models.utils import UserVars
import numpy as np
from operator import mul
from numpy.linalg import inv

import domain_creator3

################################some functions to be called#######################################
def add_atom(domain,ids=[],els=[],height=2.0):
    for i in range(len(ids)):
        domain.add_atom(ids[i],els[i],0.5,0.5,height,0.5,1.0,1.0)
#function to export refined atoms positions after fitting
def print_data(yes=False,N_sorbate=4,id='0001'):
    if yes==True:
        data=domain1A._extract_values()
        index_all=range(len(data[0]))
        index=index_all[0:20]+index_all[40:40+N_sorbate]
        f=open('/home/tlab/sphalerite/jackey/model2/files_pb/xyz_'+id+'.xyz','w')
        for i in index:
            s = '%-5s   %7.5e   %7.5e   %7.5e\n' % (data[3][i],data[0][i]*5.038,(data[1][i]-0.1391)*5.434,(data[2][i]-1.)*7.3707)
            f.write(s)
        f.close()
    else:pass     
    
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
    
def add_atom_in_slab(slab,filename):
    f=open(filename)
    lines=f.readlines()
    for line in lines:
        items=line.strip().rsplit('\t')
        slab.add_atom(items[0],items[1],float(items[2]),float(items[3]),float(items[4]),float(items[5]),float(items[6]),float(items[7]))
        
unitcell = model.UnitCell(5.038, 5.434, 7.3707, 90, 90, 90)
inst = model.Instrument(wavel = .833, alpha = 2.0)
bulk = model.Slab(T_factor='B')
domain0 =  model.Slab(c = 1.0,T_factor='B')
domain0_1 =  model.Slab(c = 1.0,T_factor='B')
rgh=UserVars()
rgh.new_var('beta', 0.0)

#add atoms for bulk and two ref domains (domain0<half layer> and domain0_1<full layer>)
#In those two reference domains, the atoms are ordered according to first hight (z values), then y values
#it is a super surface structure by stacking the surface slab on bulk slab, the repeat vector was counted 
add_atom_in_slab(bulk,'atom files in bulk.txt')
add_atom_in_slab(domain0,'atom files in domain0.txt')
add_atom_in_slab(domain0_1,'atom files in domain0_1.txt')

###########this part should be pretty much the same except for batch path head###########
#id list according to the order in the reference domain
ref_id_list=["O1_1_0","O1_2_0","Fe1_2_0","Fe1_3_0","O1_3_0","O1_4_0","Fe1_4_0","Fe1_6_0","O1_5_0","O1_6_0","O1_7_0","O1_8_0","Fe1_8_0","Fe1_9_0","O1_9_0","O1_10_0","Fe1_10_0","Fe1_12_0","O1_11_0","O1_12_0",\
"O_1_0","O_2_0","Fe_2_0","Fe_3_0","O_3_0","O_4_0","Fe_4_0","Fe_6_0","O_5_0","O_6_0","O_7_0","O_8_0","Fe_8_0","Fe_9_0","O_9_0","O_10_0","Fe_10_0","Fe_12_0","O_11_0","O_12_0"]
#the matching row Id information in the symfile
sym_file_Fe=np.array(['Fe1_0','Fe2_0','Fe3_0','Fe4_0','Fe5_0','Fe6_0','Fe7_0','Fe8_0','Fe9_0','Fe10_0','Fe11_0','Fe12_0',\
    'Fe1_1_0','Fe1_2_0','Fe1_3_0','Fe1_4_0','Fe1_5_0','Fe1_6_0','Fe1_7_0','Fe1_8_0','Fe1_9_0','Fe1_10_0','Fe1_11_0','Fe1_12_0'])
sym_file_O=np.array(['O1_0','O2_0','O3_0','O4_0','O5_0','O6_0','O7_0','O8_0','O9_0','O10_0','O11_0','O12_0',\
    'O1_1_0','O1_2_0','O1_3_0','O1_4_0','O1_5_0','O1_6_0','O1_7_0','O1_8_0','O1_9_0','O1_10_0','O1_11_0','O1_12_0'])
batch_list=['/home/jackey/apps/batchfile/','/home/jackey/model2/batchfile/','/center/w/cqiu/batchfile/','/archive/u1/uaf/cqiu/batchfile/']
batch_path_head=batch_list[1]
sym_file_head='/home/jackey/genx_data/'
       
###################create domain classes and initiate the chemical equivalent domains####################
#when change or create a new domain, make sure the terminated_layer (start from 0)set right
######################################setup domain1############################################
rgh_domain1=UserVars()
domain_class_1=domain_creator3.domain_creator(ref_domain=domain0,id_list=ref_id_list,terminated_layer=0,new_var_module=rgh_domain1)
domain1A=domain_class_1.domain_A
domain1B=domain_class_1.domain_B
domain_class_1.domain1A=domain1A
domain_class_1.domain1B=domain1B

#Adding sorbates to domain1A
add_atom(domain=domain1A,ids=['Pb1','HO_1'],els=['Pb','O'],height=2.)
#set variables
domain_class_1.set_new_vars(head_list=['u_o_n','u_Fe_n','oc_n'],N_list=[4,3,7])
domain_class_1.set_discrete_new_vars_batch(batch_path_head+'new_varial_file_standard_A.txt')
#note the grouping here is on a layer basis, ie atoms of same layer are groupped together
#you may group in symmetry, then atoms of same layer are not independent.
atm_gp_list_domain1=domain_class_1.grouping_sequence_layer(domain=[domain1A,domain1B], first_atom_id=['O1_1_0','O1_7_0'],\
    sym_file=None,id_match_in_sym={'Fe':sym_file_Fe,'O':sym_file_O},layers_N=7,use_sym=False)
domain_class_1.atm_gp_list_domain1=atm_gp_list_domain1

#you may also only want to group each chemically equivalent atom from two domains
atm_gp_discrete_list_domain1=[]
ids_domain1A=["O1_1_0","O1_2_0","Fe1_2_0","Fe1_3_0","O1_3_0","O1_4_0","Fe1_4_0","Fe1_6_0","O1_5_0","O1_6_0"]
#double check the order here
ids_domain1B=["O1_8_0","O1_7_0","Fe1_9_0","Fe1_8_0","O1_10_0","O1_9_0","Fe1_12_0","Fe1_10_0","O1_12_0","O1_11_0"]
for i in range(len(ids_domain1A)):
    atm_gp_discrete_list_domain1.append(domain_class_1.grouping_discrete_layer(domain=[domain1A,domain1B],\
domain_class_1.atm_gp_discrete_list_domain1=atm_gp_discrete_list_domain1                                                                        atom_ids=[ids_domain1A[i],ids_domain1B[i]]))

###########################setup domain2################################
#same reference domain,but set the occupancy of second iron layer to 0
rgh_domain2=UserVars()
domain_class_2=domain_creator3.domain_creator(ref_domain=domain0,id_list=ref_id_list,terminated_layer=0,new_var_module=rgh_domain2)
domain2A=domain_class_2.domain_A
domain2B=domain_class_2.domain_B
domain_class_2.domain2A=domain2A
domain_class_2.domain2B=domain2B
#set variables
domain_class_2.set_new_vars(head_list=['u_o_n','u_Fe_n','oc_n'],N_list=[4,3,7])
domain_class_2.set_discrete_new_vars_batch(batch_path_head+'new_varial_file_standard_B.txt')

atm_gp_list_domain2=domain_class_2.grouping_sequence_layer(domain=[domain2A,domain2B], first_atom_id=['O1_1_0','O1_7_0'],\
    sym_file=None,id_match_in_sym={'Fe':sym_file_Fe,'O':sym_file_O},layers_N=7,use_sym=False)
domain_class_2.atm_gp_list_domain2=atm_gp_list_domain2

#you may also only want to group each chemically equivalent atom from two domains
atm_gp_discrete_list_domain2=[]
ids_domain2A=["O1_1_0","O1_2_0","Fe1_2_0","Fe1_3_0","O1_3_0","O1_4_0","Fe1_4_0","Fe1_6_0","O1_5_0","O1_6_0"]
#double check the order here
ids_domain2B=["O1_8_0","O1_7_0","Fe1_9_0","Fe1_8_0","O1_10_0","O1_9_0","Fe1_12_0","Fe1_10_0","O1_12_0","O1_11_0"]
for i in range(len(ids_domain2A)):
    atm_gp_discrete_list_domain2.append(domain_class_2.grouping_discrete_layer(domain=[domain2A,domain2B],\
domain_class_2.atm_gp_discrete_list_domain2=atm_gp_discrete_list_domain2

def Sim(data):
    #extract the fitting par values in the associated attribute and then do the scaling(initiation+processing, actually update the fitting parameter values)
    domain_class_1.init_sim_batch2(batch_path_head+'sim_batch_file_standard_A.txt')
    domain_class_1.scale_opt_batch2b(batch_path_head+'scale_operation_file_standard_A.txt')
    #edit sim and scale file carefully for domain_class2
    domain_class_2.init_sim_batch2(batch_path_head+'sim_batch_file_standard_B.txt')
    domain_class_2.scale_opt_batch2b(batch_path_head+'scale_operation_file_standard_B.txt')
    #print refined data of atom positions
    print_data(yes=bool(0),N_sorbate=4,id='0003')
    
    #match_order to define the atoms to be considered for bond valence check
    #match_lib to define the way to calculate the bond valence
    ##############doing matching for domain1A####################
    match_order_1A=['Pb1','HO_1','O1_1_0','O1_2_0','O1_3_0','O1_4_0','O1_5_0','O1_6_0','Fe1_4_0','Fe1_6_0']
    #means consider dist btw O1_1_0 and Fe1_4_0 with no offset and dist btw 01_1_0 and Pb1 with full offset(+x,-x,...and sucha)
    match_lib_1A={'O1_1_0':create_list(ids=['Fe1_4_0','Pb1'],off_set_begin=[None],start_N=1)}
    match_lib_1A['O1_2_0']=create_list(ids=['Fe1_6_0','Pb1'],off_set_begin=['+x'],start_N=1)
    match_lib_1A['O1_3_0']=create_list(ids=['Fe1_4_0','Fe1_6_0','Pb1'],off_set_begin=[None,None],start_N=2)
    match_lib_1A['O1_4_0']=create_list(ids=['Fe1_4_0','Fe1_6_0','Pb1'],off_set_begin=['-y',None],start_N=2)
    match_lib_1A['O1_5_0']=create_list(ids=['Fe1_4_0','Fe1_6_0','Fe1_8_0','Pb1'],off_set_begin=[None,'+x','+x'],start_N=3)
    match_lib_1A['O1_6_0']=create_list(ids=['Fe1_4_0','Fe1_6_0','Fe1_9_0','Pb1'],off_set_begin=['-y',None,None],start_N=3)
    match_lib_1A['Fe1_4_0']=[['O1_1_0', 'O1_3_0','O1_5_0', 'O1_7_0', 'O1_4_0', 'O1_6_0'],[None,None,None,None,'+y','+y']]
    match_lib_1A['Fe1_6_0']=[['O1_3_0', 'O1_4_0', 'O1_8_0', 'O1_2_0', 'O1_5_0', 'O1_6_0'],[None,None,None,'-x','-x','-x']]
    match_lib_1A['Pb1']=create_list(ids=['O1_1_0','O1_2_0','O1_3_0','O1_4_0','O1_5_0','O1_6_0'],off_set_begin=[],start_N=0)
    match_lib_1A['HO_1']=[['Pb1'],[None]]
    ##############doing matching for domain2A####################
    match_order_2A=['O1_1_0','O1_2_0','O1_3_0','O1_4_0','O1_5_0','O1_6_0','Fe1_4_0','Fe1_6_0']
    #means consider dist btw O1_1_0 and Fe1_4_0 with no offset and dist btw 01_1_0 and Pb1 with full offset(+x,-x,...and sucha)
    match_lib_2A={'O1_1_0':[['Fe1_4_0'],[None]]}
    match_lib_2A['O1_2_0']=[['Fe1_6_0'],['+x']]
    match_lib_2A['O1_3_0']=[['Fe1_4_0','Fe1_6_0'],[None,None]]
    match_lib_2A['O1_4_0']=[['Fe1_4_0','Fe1_6_0'],['-y',None]]
    match_lib_2A['O1_5_0']=[['Fe1_4_0','Fe1_6_0','Fe1_8_0'],[None,'+x','+x']]
    match_lib_2A['O1_6_0']=[['Fe1_4_0','Fe1_6_0','Fe1_9_0'],['-y',None,None]]
    match_lib_2A['Fe1_4_0']=[['O1_1_0', 'O1_3_0','O1_5_0', 'O1_7_0', 'O1_4_0', 'O1_6_0'],[None,None,None,None,'+y','+y']]
    match_lib_2A['Fe1_6_0']=[['O1_3_0', 'O1_4_0', 'O1_8_0', 'O1_2_0', 'O1_5_0', 'O1_6_0'],[None,None,None,'-x','-x','-x']]

    F =[]
    beta=rgh.beta
    wt1=rgh_domain1.wt/(rgh_domain1.wt+rgh_domain2.wt)
    wt2=rgh_domain2.wt/(rgh_domain1.wt+rgh_domain2.wt)
    domain={'domain1A':{'slab':domain1A,'wt':1.}}
    sample = model.Sample(inst, bulk, domain, unitcell,coherence=False,surface_parms={'delta1':0.,'delta2':0.1391})
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
            bond_valence=domain_class_2.cal_bond_valence3(domain=domain2A,match_lib=match_lib_2A)
            t=[]
            for i in match_order_2A:
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