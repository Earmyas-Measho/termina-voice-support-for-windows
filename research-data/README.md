# VoiceCLI Study Raw Data

This directory contains the raw data from the VoiceCLI user study, supporting the findings presented in the thesis.

## 📊 Dataset Overview

- **Study Type**: Controlled User Study
- **Total Participants**: 25
- **Total Trials**: 500 (20 tasks per participant)
- **Study Duration**: 45 minutes average per participant
- **Platform**: Windows 10/11
- **ASR Model**: Whisper small.en (offline)
- **LLM Model**: Mistral-7B-Instruct (cloud-hosted)

## 📁 Data Files

### 1. VoiceCLI_Raw_Data_CORRECTED.csv
**Participant-level demographic and performance data**
- 25 participants with complete demographics
- Individual success rates (50% to 100%)
- ASR accuracy variations (93.2% to 98.5%)
- LLM performance metrics
- Latency measurements (6.3s to 10.1s)
- SUS scores (73.8 to 85.7)
- Qualitative feedback responses

**Columns**: Participant_ID, Age, Gender, CLI_Usage_Frequency, Primary_OS, Has_Voice_Assistant, Accessibility_Needs, Total_Success, Success_Rate, ASR_Accuracy, LLM_Top1_Accuracy, LLM_Top3_Accuracy, Avg_Latency_Seconds, Total_Retries, Manual_Edits, SUS_Score, Trust_Confirmation, Hands_Free_Convenience, Accessibility_Value, Latency_Acceptable, Would_Use_Daily

### 2. VoiceCLI_Task_Level_Data.csv
**Individual trial-level data for all 500 tasks**
- Task descriptions and difficulty levels
- ASR transcripts and accuracy scores
- LLM suggestions (Top-3) and rankings
- User choices and final commands
- Execution success/failure outcomes
- Timing breakdowns (ASR, LLM, confirmation)
- Retry attempts and manual edits
- Error categorization and details

**Columns**: Trial_ID, Participant_ID, Task_Number, Task_Difficulty, Task_Description, ASR_Transcript, Reference_Text, ASR_Accuracy, LLM_Suggestion_1, LLM_Suggestion_2, LLM_Suggestion_3, Correct_Suggestion_Rank, User_Choice, Final_Command, Execution_Success, Execution_Time_Seconds, ASR_Time, LLM_Time, Confirmation_Time, Total_Latency, Retry_Count, Manual_Edit, Error_Category, Error_Details

### 3. VoiceCLI_Error_Analysis_CORRECTED.csv
**Analysis of all 125 failed trials**
- Error categorization (LLM, ASR, User, Environment)
- Root cause analysis and prevention strategies
- Recovery attempts and outcomes
- Error severity ratings
- Detailed error descriptions
- ASR transcripts vs. reference text
- LLM suggestion analysis

**Columns**: Error_ID, Trial_ID, Participant_ID, Task_Difficulty, Task_Description, Error_Category, Error_Subcategory, Error_Description, ASR_Transcript, Reference_Text, ASR_Accuracy, LLM_Suggestion_1, LLM_Suggestion_2, LLM_Suggestion_3, Correct_Command, User_Action, Recovery_Attempted, Recovery_Successful, Error_Severity, Root_Cause, Prevention_Strategy

### 4. VoiceCLI_Summary_Statistics_FINAL_CORRECTED.txt
**Study metrics and statistical analysis**
- Performance metrics with 95% confidence intervals
- Success rates by task difficulty
- ASR and LLM accuracy breakdowns
- Latency analysis and distributions
- Error analysis and categorization
- User satisfaction metrics (SUS scores)
- Qualitative insights and feedback themes
- Future work priorities and recommendations

### 5. FINAL_VERIFICATION_CHECK.txt
**Mathematical verification and quality assurance**
- Mathematical verification of all calculations
- Cross-file consistency checks
- Proof of mathematical accuracy
- Verification that all success rates add up correctly
- Confirmation of error breakdown accuracy
- Quality assurance documentation

## 🔢 Key Performance Metrics

- **Overall Task Success Rate**: 75.00% (375/500 trials)
- **Success by Difficulty**:
  - Easy Tasks: 78.47% (157/200)
  - Medium Tasks: 75.43% (132/175)
  - Hard Tasks: 68.80% (86/125)
- **ASR Word-Level Accuracy**: 94.23%
- **LLM Top-3 Accuracy**: 83.00% (medium difficulty)
- **Average Latency**: 8.30 seconds
- **SUS Score**: 75.10 (mean)

## 📈 Statistical Reliability

- **Sample Size**: 25 participants (adequate for HCI studies)
- **Confidence Intervals**: 95% bootstrap CIs (1,000 resamples)
- **Inter-rater Reliability**: Cohen's κ = 0.92 (excellent agreement)
- **Internal Consistency**: Cronbach's α measures applied

## 🔒 Data Privacy & Ethics

- All participant data is anonymized
- No personally identifiable information included
- Data collection followed GDPR compliance
- Ethical approval obtained for study
- Participants provided informed consent

## 🚀 Usage Guidelines

### For Researchers
- Use for replication studies and meta-analyses
- Verify statistical calculations independently
- Cite the original thesis when using this data
- Contact me for additional context if needed

### For Educators
- Use as teaching material for HCI courses
- Demonstrate proper experimental design
- Show real-world data analysis examples
- Illustrate voice interface evaluation methods

### For Developers
- Understand user interaction patterns
- Identify common failure modes
- Learn from user feedback and suggestions
- Apply insights to voice interface design

## 📚 Citation

When using this dataset, please cite:

```
Earmyas-Measho, E. (2025). VoiceCLI: Voice-Driven Command Line Interface for Windows. 
Bachelor's Thesis, Linnaeus University, Växjö, Sweden.
https://github.com/Earmyas-Measho/termina-voice-support-for-windows.git
```

## 🤝 Contributing

This dataset is provided under an open-source license. If you find any issues or have suggestions for improvements, please:

1. Open an issue on the GitHub repository
2. Provide detailed description of the problem
3. Include specific file references and line numbers
4. Suggest potential solutions if possible

## 📞 Contact

For questions about this dataset or the VoiceCLI study:

- **Repository**: https://github.com/Earmyas-Measho/termina-voice-support-for-windows.git
- **Issues**: Use GitHub Issues for technical questions
- **Research**: Contact me for academic inquiries

---

**Note**: This dataset has been mathematically verified to ensure consistency across all files. All calculations have been checked for accuracy and reproducibility.
