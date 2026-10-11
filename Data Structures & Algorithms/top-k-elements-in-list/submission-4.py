class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        # Hash table + Bucket sort

        # Count freq
        count = {}
        for num in nums:
            count[num] = 1 + count.get(num, 0)

        # Bucket Sort
        # Store in bucket
        freq = [[] for i in range(len(nums) + 1)]
        for num, cnt in count.items():
            freq[cnt].append(num)

        # Collect
        res = []
        for i in reversed(range(len(freq))):
            for num in freq[i]: 
                res.append(num)                
            if len(res) >= k:
                return res