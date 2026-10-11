class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        # Two pointer
        res = []

        # Sort
        nums.sort()

        # two pointer scan for each a
        for i, a in enumerate(nums):
            # Min > 0 -> No answer
            if a > 0:
                break

            # pass repeated val a
            if i > 0 and a == nums[i - 1]:
                continue

            # two pointers
            l, r = i + 1, len(nums) - 1
            while l < r:
                threeSum = a + nums[l] + nums[r]

                if threeSum > 0:
                    r -= 1
                elif threeSum < 0:
                    l += 1
                else:
                    res.append([a, nums[l], nums[r]])
                    
                    # find any other pairs
                    l += 1
                    r -= 1

                    # pass repeated val nums[l]
                    while nums[l] == nums[l - 1] and l < r:
                        l += 1

        return res