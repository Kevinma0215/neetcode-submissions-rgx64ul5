class Solution:
    def maxArea(self, heights: List[int]) -> int:
        # Two pointers
        l, r = 0, len(heights) - 1
        res = 0

        # scan
        while l < r:
            area = min(heights[l], heights[r]) * (r - l)
            res = max(res, area)
            
            # move the shorter to find better one
            if heights[l] <= heights[r]:
                l += 1
            else:
                r -= 1
            
        return res
