# -*- coding: utf-8 -*-
"""
Klausur WS1920 
Methode der finiten Elemente in 2D
Netz

"""

import numpy as np
import meshtools as mt
import matplotlib.pyplot as plt

# Abmessungen
Ra=0.3
dd=0.01
Hm=0.08
Bm=0.08
ma=0.01
mb=0.05
rm0=0.095
zm0=0.
lls=0.001
sd=0.005
SH=0.01

length=0.01
epsL=length/100

p1,v1=mt.CircleSegments([0,0],Ra,a_min=-np.pi/2,a_max=np.pi/2,edge_length=length)
p2,v2=mt.LineSegments([0,Ra],[0,-Ra],edge_length=length)
p,v=mt.AddSegments(p1,p2,closed=True)
    
    
# innerer Magnetkreis
#aussen
pa,va=mt.PointSegments( [[epsL,-Hm/2.],[Bm,-Hm/2.],[Bm,-epsL]],edge_length=length/3)
p,v=mt.AddCurves(p,v,pa,va)
pa,va=mt.PointSegments( [[Bm,epsL],[Bm,Hm/2.],[epsL,Hm/2]],edge_length=length/3)
p,v=mt.AddCurves(p,v,pa,va)
# innen
pb,vb=mt.PointSegments( [[dd,-epsL],[dd,-Hm/2.+dd],[Bm-dd,-Hm/2.+dd],[Bm-dd,-epsL]],edge_length=length/3)
p,v=mt.AddCurves(p,v,pb,vb)
pb,vb=mt.PointSegments( [[Bm-dd,epsL],[Bm-dd,Hm/2.-dd],[dd,Hm/2.-dd],[dd,epsL]],edge_length=length/3)
p,v=mt.AddCurves(p,v,pb,vb)

# rAchse
pc,vc=mt.PointSegments([[epsL/2.,0],[1*epsL,0],[2*epsL,0],[3*epsL,0],[5*epsL,0],[7*epsL,0],[9*epsL,0],[dd,0],[dd+lls,0],[dd+lls+sd,0],[Bm-dd-lls-sd,0],[Bm-dd-lls,0],[Bm-dd,0],[Bm,0],[Bm+lls,0],[Bm+lls+sd,0],[Ra-epsL,0]],edge_length=length/20.)
p,v=mt.AddCurves(p,v,pc,vc)

# ende Material 1
pc,vc=mt.LineSegments([Bm-dd,Hm/2.-epsL],[Bm-dd,Hm/2.-dd+epsL],edge_length=length/3.)
p,v=mt.AddCurves(p,v,pc,vc)
pc,vc=mt.LineSegments([Bm-dd,-Hm/2+epsL],[Bm-dd,-Hm/2.+dd-epsL],edge_length=length/3.)
p,v=mt.AddCurves(p,v,pc,vc)

#Magnet
pd,vd=mt.PointSegments( [[rm0-ma/2.,-epsL],[rm0-ma/2.,zm0-mb/2.],[rm0+ma/2.,zm0-mb/2],[rm0+ma/2.,-epsL]],edge_length=length/5)
p,v=mt.AddCurves(p,v,pd,vd)
pd,vd=mt.PointSegments( [[rm0+ma/2.,epsL],[rm0+ma/2.,zm0+mb/2.],[rm0-ma/2.,zm0+mb/2],[rm0-ma/2.,epsL]],edge_length=length/5)
p,v=mt.AddCurves(p,v,pd,vd)
#Spule1
pf,vf=mt.PointSegments( [[dd+lls,-epsL],[dd+lls,-SH],[dd+lls+sd,-SH],[dd+lls+sd,-epsL]],edge_length=length/10)
p,v=mt.AddCurves(p,v,pf,vf)
pf,vf=mt.PointSegments( [[dd+lls+sd,epsL],[dd+lls+sd,SH],[dd+lls,SH],[dd+lls,epsL]],edge_length=length/10)
p,v=mt.AddCurves(p,v,pf,vf)
#Spule 1, links
pf,vf=mt.PointSegments( [[Bm-dd-lls-sd,-epsL],[Bm-dd-lls-sd,-SH/2],[Bm-dd-lls,-SH/2],[Bm-dd-lls,-epsL]],edge_length=length/10)
p,v=mt.AddCurves(p,v,pf,vf)
pf,vf=mt.PointSegments( [[Bm-dd-lls,epsL],[Bm-dd-lls,SH/2],[Bm-dd-lls-sd,SH/2],[Bm-dd-lls-sd,epsL]],edge_length=length/10)
p,v=mt.AddCurves(p,v,pf,vf)
#Spule 1, rechts
pf,vf=mt.PointSegments( [[Bm+lls,-epsL],[Bm+lls,-SH/2],[Bm+lls+sd,-SH/2],[Bm+lls+sd,-epsL]],edge_length=length/10)
p,v=mt.AddCurves(p,v,pf,vf)
pf,vf=mt.PointSegments( [[Bm+lls+sd,epsL],[Bm+lls+sd,SH/2],[Bm+lls,SH/2],[Bm+lls,epsL]],edge_length=length/10)
p,v=mt.AddCurves(p,v,pf,vf)

#Verfeinerung
def myrefine(tri_points, area):
  center_tri = np.sum(np.array(tri_points), axis=0)/3.
  r=center_tri[0]
  z=center_tri[1]
  if r<1.05*(rm0+ma/2) and -0.6*Hm<z<0.6*Hm:
    max_area=0.033*length**2
  else: 
    rho=np.sqrt(r**2+z**2)  
    max_area=70*length**2*rho
  return bool(area>max_area);
      
p,t,BouE,li_BE,bou_elem,CuE,li_CE=mt.DoTriMesh(p,v,edge_length=length,tri_refine=myrefine,writeTo='KlausurWS1920_Netz')

# read mesh
p,t,BouE,li_BE,bou_elem,CuE,li_CE=mt.LoadTriMesh('KlausurWS1920_Netz.npz')

    
# Randkurven
Ps=[[0,Ra],[0,-Ra],[0,Ra]]
bseg=mt.RetrieveSegments(p,BouE,li_BE,Ps,['Nodes','Segments'])
mt.PlotBoundary(p,bseg[0],'Nodes')
mt.PlotBoundary(p,bseg[1],'Segments')
plt.show()

# Knotenidizes zu [R0,0], [R1,0], [R2,0]
Ps=[[dd,0],[Bm-dd,0],[Bm,0]]
fnodes=mt.FindClosestNode(range(len(p)),p,Ps)
node0=fnodes[0][0]
node1=fnodes[0][1]
node2=fnodes[0][2]

# r-Achse
Ps=[ [epsL,0],[Ra-epsL,0] ]
rAchse=mt.RetrieveSegments(p,CuE,li_CE,Ps,['Nodes'])
mt.PlotBoundary(p,rAchse[0],'Nodes')
plt.show()
rAchse=rAchse[0]

print("Nodes",len(p))
print("Elements",len(t))


    
 





