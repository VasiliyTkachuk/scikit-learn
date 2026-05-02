class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        left: int = 0
        right: int = 0
        max_length: int = 0
        char_set: set[str] = set()

        while right < len(s):
            if s[right] in char_set:
                char_set.remove(s[left])
                left += 1
            else:
                char_set.add(s[right])
                right += 1
                max_length = max(max_length, right - left)

        return max_length

solution = Solution()
print(solution.lengthOfLongestSubstring("abcabcbb"))