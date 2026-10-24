class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        maxSum = nums[0]
        curSum = 0

        for i in range(len(nums)):
            # exploration
            curSum += nums[i]

            # record
            maxSum = max(maxSum, curSum)

            # shrink
            if curSum < 0:
                curSum = 0

        return maxSum