from logistic_reg import LogisticRegression
from fileread import CSV
import matplotlib.pyplot as plt

def main():
    # Define the filename with the file path name 
    filename="D:\\ML\\Logistic_Regression\\file.csv"
    csv = CSV(filename)
    csv.readFile()
    x,y=csv.getdata()
    print(x,"\n",y)

    # initialization of m,c,learning_rate and total no of iteration in epoch 
    m=0
    c=0
    learning_rate=0.0001
    epoch=1000

    reg=LogisticRegression()
    m,c=reg.gradient_descent(m,c,learning_rate,x,y,epoch)
    print(f"\nSlope(m) : {m:.5f} and Bias(c) : {c:.5f} ")

    y_predict=[reg.sigmoid(m*value+c) for value in x]

    print("\n Pass/Fali\tScore")
    for i in range(len(x)) :
        print(f"\n {y_predict[i]:.5f}\t{x[i]}")


    plt.scatter(x,y,marker="x",color="black")
    plt.plot(x,y_predict,marker="o",label="y=f(x)")
    plt.xlabel("Score(Independent variable)")
    plt.ylabel("Pass/Fail(Dependent variable)")
    plt.show()

if __name__ == "__main__":
    main()