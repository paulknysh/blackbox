import numpy as np
import csv
import multiprocessing as mp










'''
minimizes (maximizes, if applied on 1/(f+1) or so) given positive expensive black-box function, saves iterations into result file

input:
f - name of function
resfile - name of .csv file to save iterations
box - list of ranges for each variable (numpy array)
cores - number of cores available
n - number of initial function calls
it - number of subsequent function calls
tratio - fraction of initially sampled points to select threshold
rho0 - initial "balls density"
p - rate of "balls density" decay (p=1 - linear, p>1 - faster, 0<p<1 - slower)
nrand - number of random samples that are used to cover space for fit minimizing and rescaling
vf - fraction of nrand that is used for rescaling

output:
.csv file with iterations is created in the same directory

'''

def search(f,resfile,box,cores,n,it,tratio,rho0,p,nrand,vf):

	# space size
	d=len(box)

	# adjusting the number of iterations to the number of cores
	if np.mod(n,cores)!=0:
		n=n-np.mod(n,cores)+cores

	if np.mod(it,cores)!=0:
		it=it-np.mod(it,cores)+cores

	# scales a given point from a unit cube to box
	def cubetobox(pt):
		res=np.zeros(d)
		for i in range(d):
			res[i]=box[i,0]+(box[i,1]-box[i,0])*pt[i]
		return res

	# generating latin hypercube
	pts=np.zeros((n,d+1))
	lh=latin(n,d)

	for i in range(n):
		for j in range(d):
			pts[i,j]=lh[i,j]
	
	# initial sampling
	for i in range(n/cores):
		pts[cores*i:cores*(i+1),-1]=pmap(f,map(cubetobox,pts[cores*i:cores*(i+1),0:-1]),cores)

	# selecting threshold, rescaling function
	t=pts[pts[:,-1].argsort()][np.ceil(tratio*n)-1,-1]
	
	def fscale(fval):
		if fval<t:
			return fval/t
		else:
			return 1.

	for i in range(n):
		pts[i,-1]=fscale(pts[i,-1])

	# volume of d-dimensional ball (r=1)
	if np.mod(d,2)==0:
		v1=np.pi**(d/2)/np.math.factorial(d/2)
	else:
		v1=2*(4*np.pi)**((d-1)/2)*np.math.factorial((d-1)/2)/np.math.factorial(d)


	# iterations (current iteration m is equal to h*cores+i)
	T=np.identity(d)

	for h in range(it/cores):

		# refining scaling matrix T
		if d>1:

			pcafit=rbf(pts,np.identity(d))

			cover=np.zeros((nrand,d+1))
			cover[:,0:-1]=np.random.rand(nrand,d)
			for i in range(nrand):
				cover[i,-1]=pcafit(cover[i,0:-1])

			cloud=cover[cover[:,-1].argsort()][0:np.ceil(vf*nrand),0:-1]

			eigval,eigvec=np.linalg.eig(np.cov(np.transpose(cloud)))

			T=np.zeros((d,d))
			for i in range(d):
				T[i]=eigvec[:,i]/np.sqrt(eigval[i])
			T=T/np.linalg.norm(T)
		

		# sampling next batch of points
		fit=rbf(pts,T)

		pts=np.append(pts,np.zeros((cores,d+1)),axis=0)

		for i in range(cores):

			r=((rho0*((it-1.-(h*cores+i))/(it-1.))**p)/(v1*(n+(h*cores+i))))**(1./d)
			
			fitmin=1.
			for j in range(nrand):

				x=np.random.rand(d)
				ok=True

				if fit(x)<fitmin:

					for k in range(n+h*cores+i):
						if np.linalg.norm(np.subtract(x,pts[k,0:-1]))<r:
							ok=False
							break
				else:
					ok=False

				if ok==True:
					pts[n+h*cores+i,0:-1]=np.copy(x)
					fitmin=fit(x)

		pts[n+cores*h:n+cores*(h+1),-1]=map(fscale,pmap(f,map(cubetobox,pts[n+cores*h:n+cores*(h+1),0:-1]),cores))


	# saving result into external file
	extfile = open(resfile,'wb')
	wr = csv.writer(extfile, dialect='excel')
	for item in pts:
	    wr.writerow(item)










'''
builds latin hypercube

input:
n - number of points
d - size of space

output:
list of points uniformly placed in d-dimensional unit cube

'''

def latin(n,d):

	# starting with diagonal shape
	pts=np.ones((n,d))

	for i in range(n):
		pts[i]=pts[i]*i/(n-1.)

	# spread function
	def spread(p):
		s=0.
		for i in range(n):
			for j in range(n):
				if i > j:
					s=s+1./np.linalg.norm(np.subtract(p[i],p[j]))
		return s

	# minimizing spread function by shuffling
	currminspread=spread(pts)

	for m in range(1000):

		p1=np.random.randint(n)
		p2=np.random.randint(n)
		k=np.random.randint(d)

		newpts=np.copy(pts)
		newpts[p1,k],newpts[p2,k]=newpts[p2,k],newpts[p1,k]
		newspread=spread(newpts)

		if newspread<currminspread:
			pts=np.copy(newpts)
			currminspread=newspread

	return pts










'''
builds RBF-fit for given points (see Holmstrom, 2008 for details) using scaling matrix

input:
pts - list of multi-d points with corresponding values [[x1,x2,..,xd,val],...]
T - scaling matrix

output:
function that returns the value of the RBF-fit at a given point

'''

def rbf(pts,T):

	n=len(pts)
	d=len(pts[0])-1

	def phi(r):
		return r*r*r

	Phi=np.zeros((n,n))
	for i in range(n):
		for j in range(n):
			Phi[i,j]=phi(np.linalg.norm(np.dot(T,np.subtract(pts[i,0:-1],pts[j,0:-1]))))

	P=np.ones((n,d+1))
	for i in range(n):
		P[i,0:-1]=pts[i,0:-1]

	F=np.zeros(n)
	for i in range(n):
		F[i]=pts[i,-1]

	M=np.zeros((n+d+1,n+d+1))
	M[0:n,0:n]=Phi
	M[0:n,n:n+d+1]=P
	M[n:n+d+1,0:n]=np.transpose(P)

	v=np.zeros(n+d+1)
	v[0:n]=F

	sol=np.linalg.solve(M,v)

	lam=sol[0:n]
	b=sol[n:n+d]
	a=sol[n+d]

	def fit(z):
		res=0.
		for i in range(n):
			res=res+lam[i]*phi(np.linalg.norm(np.dot(T,np.subtract(z,pts[i,0:-1]))))
		res=res+np.dot(b,z)+a
		return res

	return fit










'''
maps a function on a batch of arguments in a parallel way using multiple cores 

input:
f - function
batch - list of arguments
n - number of cores

output:
list of corresponding values

'''

def pmap(f,batch,n):

	pool=mp.Pool(processes=n)
	res=pool.map(f,batch)
	pool.close()
	pool.join()

	return res










'''
# Example of usage in a main program:




from blackbox import *




def f(var):

	# perfroming expensive simulation
	# ...

	return ...




if __name__ == '__main__':

	search(f,'output.csv',

			box=np.array([[-1.,1.],[-1.,1.]]),

			cores=4,

			n=16,
			it=16,
			tratio=0.75,

			rho0=0.75,
			p=0.75,

			nrand=10000,

			vf=0.05

		)
'''