## it seemes doesn't wok

dir="C:\\Users\\Chuanpeng\\Desktop\\demo"
library(reticulate)

os<- import("os")
pd<- import("pandas")
os$chdir("C:\\Users\\Chuanpeng\\Desktop\\demo")
arb<- import("arboreto")

ex_matrix = read.csv("feed2python.csv", row.names=1, header=T)
mm= pd$DataFrame(mm)
matrix=pd$DataFrame$transpose(mm)


# matrix=pd$DataFrame(mm)
# matrix=pd$DataFrame$transpose(mm)

mm<- t(ex_matrix)
mm= r_to_py(mm)

df= read.csv("regNet_tf.csv")
tf_names= gsub("X", "G",names(df)[-1])
tf_names= r_to_py(tf_names)


## also
repl_python()

# run python as idel, but parrellal does not work
