class Solution:
    def longestConsecutive(self, nums: List[int]) -> int:
        # Hash set
        indices = set(nums)

        # Measure max
        longest = 0

        for num in indices:
            if (num - 1) in indices:
                continue
            length = 1
            while (num + length) in indices:
                length += 1
            longest = max(longest, length)
        
        return longest