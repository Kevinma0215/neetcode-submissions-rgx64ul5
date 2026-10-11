class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        # count in a hash table
        counter = {}
        for num in nums:
            counter[num] = 1 + counter.get(num, 0)

        # Bucket sort
        # create bucket
        freq = [[] for i in range(len(nums) + 1)]     # Why need +1?

        # fill bucket
        for num, cnt in counter.items():  # hash table {} can use *.item() => key, val
            freq[cnt].append(num)
        
        # collect
        res = []
        for i in range(len(freq) - 1, 0, -1):
            for num in freq[i]:
                res.append(num)
                if len(res) == k:
                    return res
