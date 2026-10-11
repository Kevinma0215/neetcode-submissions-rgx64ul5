class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        # Two Pointers
        res = []

        # Sort
        nums.sort()

        # Divide and Conquer: Assume a find b, c
        for i, a in enumerate(nums):
            # stop when a > 0
            if a > 0:
                return res
            
            # pass repeated a
            if i != 0 and a == nums[i - 1]:
                continue
            
            # Two pointer
            l, r = i+1, len(nums)-1

            while l < r:
                sum = a + nums[l] + nums[r]
                if sum < 0:
                    l += 1
                elif sum > 0:
                    r -= 1
                else:
                    res.append([a, nums[l], nums[r]])
                    l += 1
                    r -= 1

                    # pass repeated val
                    while nums[l] == nums[l - 1] and l < r:
                        l += 1

                    # left is strictly increase -> sum != 0
                    # while nums[r] == nums[r + 1] and l < r:   
                    #     r -= 1
            
        return res