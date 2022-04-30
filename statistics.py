# Programa : Criterios estadísticos con ajuste logaritmico lineal y cuadrático
# Autor : Diego Antonio Villalba Gonzalez
# Fecha : 27/04/2022 

from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import math

# Definimos funciones de utilidad

#-----MINIMOS CUADRADOS LINEALES CON INCERTIDUMBRE-------

def least_square(i_v,d_v,n):

    # Sumas de valores individuales
    s_x = sum(i_v)
    s_y = sum(d_v)

    #Suma de multiplicacion
    s_xy = 0
    for i in range(0,n):
        s_xy= s_xy + (i_v[i]*d_v[i])

    #Suma de cuadrado de x
    s_x2 = 0 
    for i in range(0,n):
        s_x2= s_x2 + (i_v[i]**2)

    #Cuadrado de la suma de x
    s2_x = s_x**2

    #Valor de delta
    de = (n*s_x2) - s2_x

    #Obtenemos valores de m y b

    m = (n*s_xy - s_x*s_y)/de
    b = (s_x2*s_y - s_xy*s_x)/de

    #----Calculo de incertidumbres----

    sqrt_y = 0

    #Calculamos suma interior
    s_ymxb = 0
    for i in range(0,n):
        s_ymxb = s_ymxb + (d_v[i]-m*i_v[i]-b)**2

    sqrt_y = math.sqrt(s_ymxb/(n-1))

    #Incertidumbre de m
    
    unc_m = sqrt_y * (math.sqrt(n/(de)))

    #Incertidumbre de b

    unc_b = sqrt_y * (math.sqrt(s_x2/(de)))

    # Desviación estandar de variable dependiente
    avg_y = np.average(d_v)
    d_y = 0
    for i in range(0,n):
        d_y= d_y + ((d_v[i]-avg_y)**2)

    std_y = math.sqrt(d_y)

    # Error estandar ya que las incertidumbres son iguales
    std_err = std_y / (math.sqrt(n))

    #Regresamos todos los valores obtenidos en forma de array:

    return [m,unc_m,b,unc_b,std_err]


#-----MINIMOS CUADRADOS PARABOLA -------
def least_square_cuad(i_v,d_v,n):
    # Sumas de valores individuales
    s_x = sum(i_v)
    s_y = sum(d_v)

    #Suma de multiplicacion
    s_xy = 0
    for i in range(0,n):
        s_xy= s_xy + (i_v[i]*d_v[i])

    #Suma de cuadrado de x
    s_x2 = 0 
    for i in range(0,n):
        s_x2= s_x2 + (i_v[i]**2)

    #Suma del cubo de x
    s_x3 = 0 
    for i in range(0,n):
        s_x3= s_x3 + (i_v[i]**3)

    #Suma de x^2 y
    s_x2y = 0
    for i in range(0,n):
        s_x2y = s_x2y + ((i_v[i]**2)*d_v[i])

    #Suma de x^4 
    s_x4 = 0
    for i in range (0,n):
        s_x4 = s_x4 + (i_v[i]**4)

    #Cuadrado de la suma de x
    s2_x = s_x**2

    #---Calculando nuestros términos tenemos
    S_xx = s_x2 - ((s2_x)/n)

    S_xy = s_xy - ((s_x*s_y)/n)
    
    S_xx2 = s_x3 - ((s_x*s_x2)/n)

    S_x2y = s_x2y - ((s_x2 * s_y)/n)

    S_x2x2 = s_x4 - ((s_x2**2)/n)


    #----Valor de delta
    de = (S_xx * S_x2x2)-(S_xx2)**2


    #Calculamos los coeficientes

    a = ((S_x2y * S_xx )-(S_xy * S_xx2))/de

    b = ((S_xy * S_x2x2) - (S_x2y*S_xx2))/de
    
    c = (s_y/n)-(b*(s_x/n))-(a*(s_x2/n))


    #---CALCULO DE INCERTIDUMBRES -----
    s_axbxc = 0

    # Calculo de variables que se repiten

    for i in range (0,n):
        s_axbxc = s_axbxc + ((d_v[i]-(a*(i_v[i]**2))-(b*i_v[i]-c))**2)

    si_y = math.sqrt((1/(n-3))*(s_axbxc))

    de_unc = (n*((s_x2*s_x4)-(s_x3)**2)) - ((s2_x*s_x4) - (s_x2)**3) + (2*s_x*s_x2*s_x3)


    # incertidumbres para cada factor 

    a_del = 0

    for k in range (0,n):
        a_del = a_del + ((k*(( (i_v[k]**2)*((n*s_x2) - s2_x) + i_v[k]*((s_x*s_x2)-(n*s_x3)) + (s_x*s_x3 - (s_x2**2) ) )/(de_unc))**2)*si_y)

    a_unc = math.sqrt(a_del)


    b_del = 0 
    for k in range (0,n):
        b_del = b_del + (k*( ((  (i_v[k]**2)*(s_x*s_x2 - n*s_x3) + i_v[k]*((n*s_x4) - (s_x2**2) )+(s_x2*s_x3-s_x*s_x4))/(de_unc) )**2 )*si_y)

    b_unc = math.sqrt(b_del)


    c_del = 0 

    for k in range (0,n):
        c_del = c_del + (k*(( ((i_v[k]**2)*(s_x*s_x3 - (s_x2**2)) + i_v[k]*(s_x2*s_x3 - s_x*s_x4) + (s_x2*s_x3 - s_x*s_x4))/(de_unc) )**2)*si_y)

    c_unc = math.sqrt(c_del)

    return [a,b,c,a_unc,b_unc,c_unc]



#-----COEFICIENTE DE CORRELACIÓN LINEAL DE PEARSON------

def pearson(i_v,d_v,n):    

    #Con numpy obtenemos los promedios
    avg_x = np.average(i_v)
    avg_y = np.average(d_v)

    #Obtenemos las desviaciones estandar de cada array
    d_x = 0
    for i in range(0,n):
        d_x= d_x + (i_v[i]-avg_x)**2
    std_x = math.sqrt(d_x)

    d_y = 0
    for i in range(0,n):
        d_y= d_y + (d_v[i]-avg_y)**2
    std_y = math.sqrt(d_y)

    #Realizamos la suma correspondiente
    s_xy = 0
    for i in range(0,n):
        s_xy= s_xy + ((i_v[i]-avg_x)*(d_v[i]-avg_y))

    #Calculamos pearson

    p = s_xy/(std_x*std_y)

    return p


#-----JI CUADRADA REDUCIDA ------

def ji_red(d_v,i_v,lst_sq,n):

    #Analizando nuestro video tenemos las incertidumbres 96 px 
    # 0.5 px  = 0.000442708 m (x)
    # 0.5 fps = 1/480 s (y)

    #Realizamos el calculo de ji cuadrada
    ji = 0
    m = lst_sq[0]
    b = lst_sq[2]

    # Consideramos el caso en que las incertidumbres son las mismas, 
    # en caso de lo contrario estas se deben de leer del txt y crear un array que pueda ser leido de manera analoga
    for i in range(0,n):
        ji = ji + ((d_v[i]-(m*i_v[i])-b)**2)/((0.2083)**2)

    # Calculamos nuestro numero de grados de libertad para un modelo lineal (m,b) parametros

    ngl = n-2

    ji_red = ji/ngl

    return ji_red


#-----Ji cuadrada reducida para cuadráticas ----

def ji_red_quad(d_v,i_v,a,b,c,n):

    #Analizando nuestro video tenemos las incertidumbres 96 px 
    # 0.5 px  = 0.000442708 m (x)
    # 0.5 fps = 1/480 s (y)

    #Realizamos el calculo de ji cuadrada
    ji = 0

    # Consideramos el caso en que las incertidumbres son las mismas

    for i in range(0,n):
        ji = ji + ((d_v[i]-(a*(i_v[i]**2))+(b*i_v[i])+c)**2)/((2.083)**2)

    # Calculamos nuestro numero de grados de libertad para un modelo cuadratico (a,b,c) parametros
    ngl = n-3

    ji_red = ji/ngl

    return ji_red
    
#-------   Inicio del programa -------  

#Obtenemos nuetra información del archivo txt

i_v = [] # Array para variable independiente (x)
d_v = [] # Array para variable dependiente  (y)
n=0

muestra = input("Muestra a analizar -> ")

  

with open('videos_2/video_'+muestra+'.txt', 'r') as data: 
    #leemos el numero de lineas
    lines = data.readlines() 

    #Leemos el titulo y el nombre de nuestras variables
    chart_n = lines[0]
    names = lines[1].split('\t')
    n_i_v = names[0]
    n_d_v = names[1]

    #Leemos archivo y guardamos datos en array correspondiente
    for i in range(2,len(lines)):  #saltamos encabezado
            column = lines[i].split('\t') 
            d_v.append(float(column[0])) 
            i_v.append(float(column[1])) 
            n = n+1  

#Preguntamos si tiene desfase de distancia

defas = input("Tiene desfase de variable independiente?   (y/n)   ->  ")

#Si esta desfazado realizamos la resta
if defas == "y":
    desfa = i_v[0]
    i_v = np.array(i_v)
    i_v = i_v - desfa

print("\nAjuste realizado con éxito !\n")

# Agregamos la evaluacion como logaritmo de nuestros datos
nlog = input("Realizar ajuste logaritmico (y/n)   ->  ")
if defas == "y":
    n = n-1
    del d_v[0]
    i_v = np.delete(i_v,0)
if nlog == "y" :
    d_v = np.log(d_v)
    i_v = np.log(i_v)

#Agregamos Seleccionamos la opción demínimos cuadrados a realizar 
lst_sq_kind = input("Ajuste lineal o parabólico?  (l/p)   ->  ")

#---- Ajsute lineal -------

if lst_sq_kind == "l":

    #Aplicamos nuestra función de mínimos cuadrados

    lst_sq =least_square(i_v,d_v,n)
    m= lst_sq[0]
    b= lst_sq[2]

    #Aplicamos nuestra función de coeficiente de pearson

    p = pearson(i_v,d_v,n)

    #Aplicamos nuestra funcion de  ji cuadrada reducida

    ji_r = ji_red(d_v,i_v,lst_sq,n)

    #Mosramos nuestros resultados en una grafica por medio de matplotlib

    domain = np.linspace(np.min(i_v),np.max(i_v))
    plt.plot(domain, m*domain+b,label='Ajuste por mínimos cuadrados')
    plt.title(chart_n)
    plt.xlabel(n_i_v)
    plt.ylabel(n_d_v)
    plt.scatter(i_v,d_v,label='Datos',color='green',marker='+')
    plt.legend()
    plt.show()
    # plt.savefig('resultados/plot_'+chart_n+'.png')
    # plt.close()

    # Mostramos nuestros resultados de manera individual en consola
    print("Ajuste por mínimos cuadrados: \n")
    print("m =",m," ± ", lst_sq[1],"\n")
    print("b =",b," ± ", lst_sq[3],"\n \n")

    print("Coeficiente de correlación lineal de Pearson\n")
    print("p = ",p,"\n \n")

    print("χ cuadrada reducida\n")
    print("χ_red = ",ji_r)

    #Guardamos los resultados en un archivo txt

    f = open("resultados/results_"+chart_n+".txt", "a")
    f.write("\n\n--------------------------------------------\n")
    f.write("Análisis de  -> " + chart_n + "\n")
    f.write("Fecha ->  " + str(datetime.now()) + "\n")
    f.write("--------------------------------------------\n\n")
    if nlog == "y":
        f.write("\n-------- AJUSTE LOGARÍTIMICO ----------\n")
    f.write("----- Ajuste por mínimos cuadrados -----\n")
    f.write("m ="+str(m)+" ± "+str(lst_sq[1])+"\n")
    f.write("b ="+str(b)+" ± "+ str(lst_sq[3])+"\n \n")
    f.write("----- Coeficiente de correlación lineal de Pearson -----\n")
    f.write("p = "+str(p)+"\n \n")
    f.write("----- χ cuadrada reducida -----\n")
    f.write("χ_red = "+str(ji_r))
    f.close()



#----Ajuste parabolico
if lst_sq_kind == "p":

    # Se presenta una diferencia en la variable independiente
    defas_2 = input("Tiene desfase de variable independiente?   (y/n)   ->  ")

    #Si esta desfazado realizamos la resta
    if defas_2 == "y":
        desfac = d_v[0]
        d_v = np.array(d_v)
        d_v = d_v - desfac
    
    # Aplicamos minios cuadrados
    lst_sq = least_square_cuad(d_v,i_v,n)
    a = lst_sq[0]
    b = lst_sq[1]
    c = lst_sq[2]
    a_unc = lst_sq[3]
    b_unc = lst_sq[4]
    c_unc = lst_sq[5]

    #Conocemos ji cuadrada recucida
    ji_red = ji_red_quad(d_v,i_v,a,b,c,n)

    #--Graficamos nuestros resultados
    d = np.linspace(np.min(d_v),np.max(d_v))
    plt.plot(d, a*(d**2)+b*d +c,label='Ajuste por mínimos cuadrados')
    #plt.plot(d, (-32.841)*(d**2)+ 7.215*b + 0.023, label='parabola predicha')
    plt.title(chart_n)
    plt.xlabel(n_i_v)
    plt.ylabel(n_d_v)
    plt.scatter(d_v,i_v,label='Datos',color='green',marker='+')
    plt.legend()
    plt.show()

    # Mostramos nuestros resultados de manera individual en consola
    print("Ajuste por mínimos cuadrados: \n")
    print("a =",a," ± ",a_unc,"\n")
    print("b =",b," ± ",b_unc,"\n")
    print("c =",c," ± ",c_unc,"\n \n")

    print("χ cuadrada reducida\n")
    print("χ_red = ",ji_red)