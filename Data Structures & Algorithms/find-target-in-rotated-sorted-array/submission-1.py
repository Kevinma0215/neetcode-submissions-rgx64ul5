class Solution:
    def search(self, nums: List[int], target: int) -> int:
        # binary search

        lo, hi = 0, len(nums) - 1

        while lo <= hi:
            mid = (lo + hi) // 2

            if nums[mid] == target:
                return mid
            
            if nums[lo] <= nums[mid]:
                if target > nums[mid] or target < nums[lo]:
                    lo = mid + 1
                else:
                    hi = mid - 1

            else:
                if target < nums[mid] or target > nums[hi]:
                    hi = mid - 1
                else:
                    lo = mid + 1

        return -1