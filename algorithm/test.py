def kClosest(points, K):
    distance={}
    for i in range(len(points)):
        # print(points[i][0]**2,points[i][1]**2,points[i][0]**2+points[i][1]**2)
        distance[i] = points[i][0]**2+points[i][1]**2
    print(distance)
    s=sorted(distance.items(), key=lambda x: x[1])
    # print(s[K-1][0])
    # t=s[K-1][0]
    ans=[]
    for m in range(1,K+1):
        print(m)
        print(points[s[m - 1][0]])
        ans.append(points[s[m - 1][0]])

    print(ans)
    return ans
    # print(points[s[K-1][0]])
    return points[s[K-1][0]]
points = [[1,3],[-2,2]]
K = 1
# points = [[3,3],[5,-1],[-2,4]]
# K = 2
kClosest(points, K)
