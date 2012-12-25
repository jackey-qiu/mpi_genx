import sys
genx_path='C:\\apps\\genx'
sys.path.insert(0,genx_path)
import os
import numpy as np
import model
import data
import parameters
import fom_funcs
#absolute path for script file, table file and data file
RAW_DATA='D:\\Programming codes\\modeling codes\\gx file components\\datasets\\raw_data.dat.txt'
NEW_DATA='D:\\Programming codes\\modeling codes\\gx file components\\datasets\\new_data_test.dat'
SCRIPT='D:\\Programming codes\\modeling codes\\gx file components\\scripts\\single_domain_halflayer_1pb.py'
TABLE='D:\\Programming codes\\modeling codes\\gx file components\\grid_talbes\\1pb_1OH.txt'
GENX='D:\\genx_test.gx'
class genx_file_creator:
    def __init__(self,raw_data=RAW_DATA,new_data=NEW_DATA,script_path=SCRIPT,table_path=TABLE,gx_file_path=GENX,extra_tag=[[[10,10],[10,2],0.1],[[11,11],[6,2],0.1]]):
        self.raw_data=raw_data
        self.new_data=new_data
        self.script_path=script_path
        self.table_path=table_path
        self.gx_file_path=gx_file_path
        self.extra_tag=extra_tag
        self.append_extra_dataset()
        self.create_gx_1()
        
    def append_extra_dataset(self):
        data=np.loadtxt(self.raw_data)
        for tag in self.extra_tag:
            for i in range(np.sum(tag[1])):
                if i<tag[1][0]:
                    newline=np.array([[tag[0][0],tag[0][1],i,2.,tag[2],0.,0.]])
                    data=np.append(data,newline,axis=0)
                else:
                    newline=np.array([[tag[0][0],tag[0][1],i,3.,tag[2],0.,0.]])
                    data=np.append(data,newline,axis=0)
        np.savetxt(self.new_data,data)
    
    def split_dataset(self):
        '''
        Function to split a full dataset into sub sets based on different hk values
        Data format is the one defined in gsecars_ctr.py data loader as follows:
            1st column h values; 2nd column k values; 3rd values l values;
            4th column Intensites; 5th column The standard deviation of the intensities;
            6th column L of first Bragg peak, 7th column L spacing of Bragg peaks
        Data has been sorted according to h k values
        Each sub dataset will be saved and named as hkL.dat (eg, 10L.dat, 00L.dat)
        The return is a list of absolute path of the subsets and will be called in function create_gx_1
        '''
        try:
            datapath='\\'.join(self.new_data.rsplit('\\')[0:-1])+'\\'
        except:
            datapath='\\'.join(self.new_data.rsplit('//')[0:-1])+'\\'
        name_list=[]
        data=np.loadtxt(self.new_data)
        tracer=[int(data[0,0]),int(data[0,1])]
        boundary=[0]
        for i in range(len(data)):
            if (int(data[i,0])!=tracer[0]) | (int(data[i,1])!=tracer[1]):
                tracer[0],tracer[1]=int(data[i,0]),int(data[i,1])
                boundary.append(i)
        boundary.append(len(data))
        for i in range(len(boundary)-1):
            np.savetxt(datapath+"%i%iL.dat"%(data[boundary[i],0],data[boundary[i],1]),data[boundary[i]:boundary[i+1]])
            name_list.append(os.path.abspath(datapath+"%i%iL.dat"%(data[boundary[i],0],data[boundary[i],1])))
        return name_list

    def create_many_gx_1(self,datapath="",datafile_list=['data1.dat','data2.dat'],table_file_path_head="",tab_list=['table1.tab','table2.tab'],script_file_path_head="",script_list=[],save_file_head='',save_file_list=[]):
        #call this function if you only have full dataset
        for i in range(len(tab_list)):
            name_list=self.split_dataset(datapath,datafile_list[i])
            data_list=data.DataList()
            data_list.add_new_list(name_list)
            
            table_file_path=table_file_path_head+tab_list[i]
            script_file_path=script_file_path_head+script_list[i]
            save_file=save_file_head+save_file_list[i]
            
            parameter=parameters.Parameters()
            parameter.set_ascii_input(open(table_file_path).read())
            script=unicode(open(script_file_path).read())
            mod=model.Model()
            mod.data=data_list
            mod.script=script
            mod.parameters=parameter
            mod.save(save_file)
        
    def create_gx_1(self):
        #call this function if you only have full dataset, and create only one gx files
        name_list=self.split_dataset()
        data_list=data.DataList()
        data_list.add_new_list(name_list)
        parameter=parameters.Parameters()
        parameter.set_ascii_input(open(self.table_path).read())
        script=unicode(open(self.script_path).read())
        mod=model.Model()
        mod.data=data_list
        mod.script=script
        mod.parameters=parameter
        mod.save(self.gx_file_path)
        
    def create_gx_2(self,name_list=[],table_file_path='table_file',script_file_path='script_file',save_file='D:\\test_genx.gx'):
        #call this function if you already have sub datasets, note the name_list is list of absolute path to each sub dataset
        data_list=data.DataList()
        data_list.add_new_list(name_list)
        parameter=parameters.Parameters()
        parameter.set_ascii_input(open(table_file_path).read())
        script=unicode(open(script_file_path).read())
        mod=model.Model()
        mod.data=data_list
        mod.script=script
        mod.parameters=parameter
        mod.save(save_file)
    
if __name__=="__main__":
    create_gx_1()