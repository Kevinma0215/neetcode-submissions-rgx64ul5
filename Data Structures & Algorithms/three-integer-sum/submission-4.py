class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        # Two pointer
        # sort 
        nums.sort()
        res = []

        for i in range(len(nums)):
            # check nums[i] negative
            if nums[i] > 0: break

            # check nums[i] not repeat
            if i > 0 and nums[i] == nums[i - 1]: continue

            l = i + 1
            r = len(nums) - 1

            while l < r:
                threeSum = nums[i] + nums[l] + nums[r]

                if threeSum > 0:
                    r -= 1
                elif threeSum < 0:
                    l += 1
                else:
                    res.append([nums[i], nums[l], nums[r]])
                    r -= 1
                    l += 1
                    # check nums[l] repeated
                    while l < r and nums[l] == nums[l - 1]:
                        l += 1

        return res