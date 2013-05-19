import numpy as np
from numpy.linalg import inv
#we consider the situation of known apex to calculate trigonal pyramid geometry

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
#refer the the associated ppt file when read the comments

class find_sorbate_at_right_angle():
#to add sorbate with most possible steric feature, here p0, p1 ref_p, and calculated sorbate are on the same plane, sorbate has same distance to p0 and p1
#so usually set the ref_p to be the coors of Fe inside a octahedral. 
    def __init__(self,p0=[0.,0.,0.],p1=[2.,2.,2.],ref_p=[1.,2.,3.],edge_len=3,mirror=False,use_ref=True):
        self.p0,self.p1,self.ref_p=np.array(p0),np.array(p1),np.array(ref_p)
        self.edge_len=edge_len
        if use_ref:
            self.find_position(mirror)
        else:
            self.find_position_same_level()
        
    def find_position(self,mirror=False):
        dist_p0_p1=f2(self.p0,self.p1)
        biset_len=(self.edge_len**2-(0.5*dist_p0_p1)**2)**0.5
        z_v=f3(np.zeros(3),np.cross(self.p1-(self.p0+self.p1)/2.,self.ref_p-(self.p0+self.p1)/2.))
        x_v=f3(np.zeros(3),self.p1-(self.p0+self.p1)/2.)
        y_v=np.cross(z_v,x_v)
        if mirror==True:y_v=-y_v
        self.sorbate=biset_len*y_v+(self.p0+self.p1)/2.
       
    def find_position_same_level(self):
        dist_p0_p1=f2(self.p0,self.p1)
        height=(self.edge_len**2-(0.5*dist_p0_p1)**2)**0.5
        self.sorbate=(self.p0+self.p1)/2.+[0.,0.,height]
        
    def print_file(self,file):
        f=open(file,'w')
        s = '%-5s   %7.5e   %7.5e   %7.5e\n' % ('Pb', self.sorbate[0],self.sorbate[1],self.sorbate[2])
        f.write(s)
        s = '%-5s   %7.5e   %7.5e   %7.5e\n' % ('O', self.p0[0],self.p0[1],self.p0[2])
        f.write(s)
        s = '%-5s   %7.5e   %7.5e   %7.5e\n' % ('O', self.p1[0],self.p1[1],self.p1[2])
        f.write(s)
        s = '%-5s   %7.5e   %7.5e   %7.5e\n' % ('Fe', self.ref_p[0],self.ref_p[1],self.ref_p[2])
        f.write(s)
        f.close()  
        
class trigonal_pyramid_two_point():
#if we know to points with one being sorbate position and the other is oxygen
    def __init__(self,apex=[0.,0.,0.],p0=[2.,2.,2.],top_angle=1.0,phi=0.,mirror=False):
        #top angle is p0_A_p1 in ppt file, shoulder_angle is A_P0_CP
        #phi can be value from [0,2pi]
        self.top_angle,self.phi=top_angle,phi
        self.p0,self.apex=np.array(p0),np.array(apex)
        self.cal_bottom_angle()
        self.find_position(mirror)
    
    def cal_bottom_angle(self):
        self.edge=f2(self.p0,self.apex)
        self.shoulder=self.edge*np.sin(self.top_angle/2.)*2.
        self.bottom_shoulder=self.edge*np.sin(np.pi-self.top_angle)
        self.bottom_angle=np.arcsin(self.shoulder/2./self.bottom_shoulder)*2.
        
    def find_position(self,mirror=False):
        n_v=self.p0-self.apex
        a,b,c=n_v[0],n_v[1],n_v[2]
        x0,y0,z0=self.apex[0],self.apex[1],self.apex[2]
        ref_p=0
        if c!=0.:
            ref_p=np.array([1.,1.,(a*(x0-1.)+b*(y0-1.))/c+z0])
        elif b!=0.:
            ref_p=np.array([1.,(a*(x0-1.)+c*(z0-1.))/b+y0,1.])
        else:
            ref_p=np.array([(b*(y0-1.)+c*(z0-1.))/a+x0,1.,1.])
        z_v=f3(np.zeros(3),(self.p0-self.apex))
        x_v=f3(np.zeros(3),(ref_p-self.apex))
        y_v=np.cross(z_v,x_v)
        T=f1(x0_v,y0_v,z0_v,x_v,y_v,z_v)
        theta=self.top_angle             
        phi_p1=self.phi
        phi_p2=self.phi
        if mirror==True:phi_p2=phi_p2+self.bottom_angle
        else:phi_p2=phi_p2-self.bottom_angle
        r0=self.edge
        p1_new = np.array([r0*np.cos(phi_p1)*np.sin(theta),r0*np.sin(phi_p1)*np.sin(theta),r0*np.cos(theta)])
        p2_new = np.array([r0*np.cos(phi_p2)*np.sin(theta),r0*np.sin(phi_p2)*np.sin(theta),r0*np.cos(theta)])
        self.p1=np.dot(inv(T),p1_new)+self.apex
        self.p2=np.dot(inv(T),p2_new)+self.apex
        
    def print_file(self,file):
        f=open(file,'w')
        s = '%-5s   %7.5e   %7.5e   %7.5e\n' % ('Pb', self.apex[0],self.apex[1],self.apex[2])
        f.write(s)
        s = '%-5s   %7.5e   %7.5e   %7.5e\n' % ('O', self.p0[0],self.p0[1],self.p0[2])
        f.write(s)
        s = '%-5s   %7.5e   %7.5e   %7.5e\n' % ('O', self.p1[0],self.p1[1],self.p1[2])
        f.write(s)
        s = '%-5s   %7.5e   %7.5e   %7.5e\n' % ('O', self.p2[0],self.p2[1],self.p2[2])
        f.write(s)
        f.close()

class trigonal_pyramid_three_point():
    #if we know three positions with one sorbate at apex and the other two oxygen positions
    def __init__(self,apex=[2.,0.,0.],p0=[0.,2.,0.],p1=[0.,0.,2.]):
        
        self.p0,self.apex,self.p1=np.array(p0),np.array(apex),np.array(p1)
        self.edge=f2(self.apex,self.p0)
        self.shoulder=f2(self.p0,self.p1)
        self.top_angle=np.arcsin(self.shoulder/2./self.edge)*2.
        
    def find_position(self,mirror=False):
        edge_ct=(self.p0+self.p1)/2.
        z_v=f3(np.zeros(3),(self.p0-edge_ct))
        x_v=f3(np.zeros(3),(self.apex-edge_ct))
        y_v=np.cross(z_v,x_v)
        T=f1(x0_v,y0_v,z0_v,x_v,y_v,z_v)
        theta=np.pi/2.
        biset_line_top=f2(edge_ct,self.apex)
        biset_line_bottom=self.shoulder*np.sin(np.pi/3.)
        phi=np.arccos((biset_line_top**2+biset_line_bottom**2-self.edge**2)/2./biset_line_top/biset_line_bottom)
        if mirror==True:phi=-phi
        r0=biset_line_bottom
        p2_new = np.array([r0*np.cos(phi)*np.sin(theta),r0*np.sin(phi)*np.sin(theta),r0*np.cos(theta)])
        self.p2=np.dot(inv(T),p2_new)+edge_ct
               
        
    def print_file(self,file):
        f=open(file,'w')
        s = '%-5s   %7.5e   %7.5e   %7.5e\n' % ('Pb', self.apex[0],self.apex[1],self.apex[2])
        f.write(s)
        s = '%-5s   %7.5e   %7.5e   %7.5e\n' % ('O', self.p0[0],self.p0[1],self.p0[2])
        f.write(s)
        s = '%-5s   %7.5e   %7.5e   %7.5e\n' % ('O', self.p1[0],self.p1[1],self.p1[2])
        f.write(s)
        s = '%-5s   %7.5e   %7.5e   %7.5e\n' % ('O', self.p2[0],self.p2[1],self.p2[2])
        f.write(s)
        f.close()