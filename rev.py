n=int(input("ENTER NUMBER"))
rev=0
while(n>0):
    dig=n%10
    rev=rev*10+dig
    n=n//1
print("REVERSE OF THE NUMBER:",rev)