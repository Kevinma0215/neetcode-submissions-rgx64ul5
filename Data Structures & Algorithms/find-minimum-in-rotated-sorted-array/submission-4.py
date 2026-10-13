class Solution:
    def findMin(self, nums: List[int]) -> int:
        # Binary search to find gap
        l, r = 0, len(nums) - 1

        while l < r:
            # if nums[l] < nums[r]:
            #     res = min(res, nums[l])
            #     break
            
            m = (l + r) // 2
            
            # res = min(res, nums[m])

            if nums[m] < nums[r]:
                r = m
            else: 
                l = m + 1
            
        return nums[l]