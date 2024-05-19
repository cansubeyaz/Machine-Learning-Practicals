##
i = 1
i = i + 2 ## or i +=2 ------ i /=7
print(i)

## LISTS []

color = ["red", "blue", "green"]
print(type(color))
print((len(color))) # how many attribute in the list
print(color[1:])
color.append("gray")
print(color)
color.insert(1, "magenta")
print(color)
color.remove("red")
print(color)

numb = [1, 2 , 39, 3, 8, 55]
print(max(numb))

for i1 in color:
    print(i1)

print(list(enumerate(color)))
print(list(enumerate(color, start=1)))

stringcolor = "-".join(color)
print(stringcolor)

## TUPLES()(SIMILAR LISTS) AND SETS{unordered}
## to create
bosliste1 = []
bosliste2 = list()

bosdemet1 = ()
bosdemet2 = tuple()

boskume1 = set() ## boskume = {} kume yaratmaz dictionary yaratır.
demet = ("sari", "mavi", "yesil", "gri")

python = set("PYTHON")
print(python)

## DICTIONARY

dict = {"isim" : "Ali", "yas" : 20, "cinsiyet" : "m", "hobiler" : ["sinema", "konser", "yazılım"]}

dict["id"] = 1
print(dict)

del dict["id"]

for x in dict:
    print(dict[x]) ## values of the dict

print(dict.keys())
print(dict.values())
print(dict.items())

for k in dict.items():
    print(k)

## if and else

a = 10
if a==3:
    print("a!=3")
elif a == 4:
    print("a != 4")
elif a == 5:
    print("a !=5")
else:
    print("noone")

## for and while



