# NOVA_RL: Migrate to Gemini 2.5 Flash Only
Plan approved. Steps to complete:

## 1. [x] Create TODO.md (current)
## 2. [x] Edit inference.py: Remove all Sarvam code, default to Gemini 2.5-flash
## 3. [x] Verify no other Sarvam refs (already done via search_files)
## 4. [x] Test run: python inference.py (check Gemini usage, no errors)
## 5. [x] Final verification: search_files (?i)sarvam -> 0 matches
## 6. [ ] attempt_completion
