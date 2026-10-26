class Solution:
    def minEatingSpeed(self, piles: List[int], h: int) -> int:
        # binary search
        minSpeed = max(piles)
        
        l, r = 1, minSpeed
        while l < r:
            mid = (l + r) // 2

            time = 0
            for p in piles:
                # time += math.ceil(float(p) / mid)
                time += (p + mid - 1) // mid

            if time > h:
                l = mid + 1
            else:
                r = mid
                minSpeed = mid

        return minSpeed
        
