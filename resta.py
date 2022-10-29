c = 0


a = int(input('ingrese un numero'))
b = int(input('ingrese un numero'))

if a<b:
    for i in range(a,b):
        c = c+1
    print('Resultado de la resta: -',c)
elif a>b:
    for i in range(b,a):
        c +=1
    print('resultado de la resta: ', c)
else:
    print('resta = 0')
