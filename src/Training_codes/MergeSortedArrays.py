def mergeArrays(arr1, arr2, n1, n2):
   arr3 = [None] * (n1+n2)
   i = 0;
   j = 0;
   k = 0;

   while i < n1 and j < n1:
     if arr1[i] < arr2[j]:
         arr3[k] = arr1[i]
         i = i + 1;
         k = k + 1;

     if arr1[i] > arr2[j]:
         arr3[k] = arr2[j]
         j = j + 1;
         k = k + 1;

   while i < n1:
      arr3[k] = arr1[i]
      i = i + 1;
      k = k + 1;

   while j < n2:
      arr3[k] = arr2[j]
      j = j + 1;
      k = k + 1;   
   print("Array after merging") 
   for i in range(n1 + n2): 
        print(str(arr3[i]), end = " ") 

arr1 = [0, 3, 4, 31] 
n1 = len(arr1) 
  
arr2 = [4, 6, 30] 
n2 = len(arr2) 
mergeArrays(arr1, arr2, n1, n2)
