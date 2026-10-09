class Solution:
    def hasDuplicate(self, nums: List[int]) -> bool:
        seen = set()

        for num in nums:
            if num in seen: # in set -> find it
                return True
            seen.add(num)   # set.add -> append
        
        return False
