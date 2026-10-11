class Solution:
    def longestConsecutive(self, nums: List[int]) -> int:
        # Hash set
        mp = set(nums)
        longest = 0

        # Scan and find min in a consecutive subarray
        for num in mp:  # loop in the set is better, since no duplicate
            if (num - 1) in mp:
                continue
            
            length = 1
            while (num + length) in mp:
                length += 1

            # update max
            longest = max(longest, length) 
        
        return longest