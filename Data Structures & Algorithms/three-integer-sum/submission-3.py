class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        # Two pointers
        res = []

        # Sort
        nums.sort()

        # Scan
        for i in range(len(nums) - 2):
            # check the first item
            if nums[i] > 0: break  

            # check repeat
            if i != 0 and nums[i] == nums[i - 1]: continue

            l = i + 1
            r = len(nums) - 1

            # update pointer
            while l < r:
                threeSum = nums[i] + nums[l] + nums[r]
                
                if threeSum > 0:
                    r -= 1
                elif threeSum < 0:
                    l += 1
                else:
                    res.append([nums[i], nums[l], nums[r]])
                    # update pointer
                    r -= 1
                    l += 1
                    # check repeat
                    while l < r and nums[l] == nums[l - 1]:
                        l += 1

        return res