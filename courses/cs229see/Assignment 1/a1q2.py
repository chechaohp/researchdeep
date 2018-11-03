import numpy as np

def loaddata():
    x = np.loadtxt('data/x.dat')
    y = np.loadtxt('data/y.dat')
    return x,y

def cost(X,y,theta,ld,w):
    y_pre = hypothesis_function(X,theta)
    regulation = -ld/2*np.matmul(np.transpose(theta),theta)
    lost = np.mean(w*(y*np.log(y_pre)+(1-y)*np.log(1-y_pre)))
    return lost+regulation

def grad_cost(X,y,theta,w,ld):
    z = y-hypothesis_function(X,theta)
    return np.matmul(np.transpose(X),z)-ld*theta

def Hessian(X,theta,w,ld):
    y_pre = hypothesis_function(X,theta)
    D = np.diag(-(weights*y_pre*(1-y_pre))[:,0])
    return np.matmul(np.transpose(X),np.matmul(D,X))-ld

def hypothesis_function(X,theta):
    linear = np.matmul(X,theta)
    return 1/(1+np.exp(-linear))

X_train,y_train = loaddata()
y_train = y_train.reshape(-1,1)
n_features = X_train.shape[1]
m_training = X_train.shape[0]
theta = np.zeros([n_features,1],dtype = np.float32)
ld = 0.0001
x = np.asarray([[0.05,0.1]])
tau = 0.01
print(n_features)
weights = np.exp(-np.sum(np.abs(x-X_train)**2,axis=-1)**(1./2)/(2*tau)**2).reshape(-1,1)
y_pre = hypothesis_function(X_train,theta)

for i in range(0,100):
    g = grad_cost(X_train,y_train,theta,weights,ld)
    # print(g.shape)
    H = Hessian(X_train,theta,weights,ld)
    theta = theta - np.matmul(np.linalg.pinv(H),g)
print(hypothesis_function(x,theta))