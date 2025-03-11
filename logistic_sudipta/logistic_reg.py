import math 
class LogisticRegression:

    # Implement sigmoid function 
    def sigmoid(self,z):
        if z > 500:return 1.0 # to prevent overflow condition
        elif z <-500 :return 0.0 # to prevent underflow condition
        return (1/(1+math.exp(-z)))
    

    # implement Cost Function (to check the error)
    # Means how much error exists b/w actual value and predicted value  
    def cost_function(self,m,c,x,y):
        total_cost=0
        epsilon=1e-9 # a small constant to avoid log(0) error
        for i in range(len(x)):
            # y predicted value 
            hx=self.sigmoid(m*x[i]+c)
            hx=min(max(hx,epsilon),1-epsilon) # it tendes to 0 but never can be 0
            #cost calculation using formula //cost = 1/len(x) i=1 to len(x) sum(-y*log(h(x))-(1-y)*log(1-h(X)))
            total_cost=total_cost+(-y[i]*math.log(hx)-(1-y[i])*math.log(1-hx))

        return total_cost/len(x) #Average cost calculation 
    

    #Implement gradient descent function to optimize m and c (to find out global minima)
    def gradient_descent(self,m,c,learning_rate,x,y,epoch):

        # total gradient w.r.t.m
        total_dm=0
        # total gradient w.r.t.c
        total_dc=0

        for k in range(epoch):
            if k%100 == 0 :
                print(f"epoch = {k} and cost : {self.cost_function(m,c,x,y):.5f}")

            for i in range(len(x)):
                y_predict=self.sigmoid(m*x[i]+c) # y_predict = h(x)=1/1+e^-(m*X[i]+c)
                error=y_predict-y[i]  #error = (h(X)i-Yi)

                total_dm=total_dm+error*x[i] # store inside total_dm = partial derivative w.r.t.m = error * x[i]
                total_dc=total_dc+error      # store inside total_dc = partial derivative w.r.t.c =  error

            m=m-learning_rate*total_dm
            c=c-learning_rate*total_dc
        return m,c
