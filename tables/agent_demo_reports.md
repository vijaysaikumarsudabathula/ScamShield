# ScamShield agent - demo reports

### HIGH - likely scam  (risk 1.00)
> Your account has been temporarily suspended due to unusual activity. Verify your identity within 24 hours: http://secure-bank-verify.xyz/login

- Text model P(scam): **0.999**; domain risk: **0.982**
- Tactics: Urgency / time pressure, Authority impersonation, Fear / threat, Payment / credential request
- Evidence: secure-bank-verify.xyz stuffs security/finance words into the address (verify, secure, bank); secure-bank-verify.xyz uses many hyphens, a common phishing pattern; '.xyz' is a low-cost TLD heavily used by phishing sites; domain model rates secure-bank-verify.xyz as phishing-like (P=0.98); manipulation tactics: Urgency / time pressure, Authority impersonation, Fear / threat, Payment / credential request
- Top contributing terms: contains a link, word 'your', word 'http', word 'to', word 'been'
- Advice: Do not click any link, call any number or reply. Never share OTPs, PINs, passwords or card details. Contact the organisation only via its official app or the number printed on your card/bill.

### HIGH - likely scam  (risk 1.00)
> Your package could not be delivered due to an incorrect address. Pay a $1.99 redelivery fee: http://postal-redeliver.info/pay

- Text model P(scam): **1.000**; domain risk: **0.809**
- Tactics: Payment / credential request
- Evidence: '.info' is a low-cost TLD heavily used by phishing sites; domain model rates postal-redeliver.info as phishing-like (P=0.81); manipulation tactics: Payment / credential request
- Top contributing terms: contains a link, mentions money, word 'your', word 'info', word 'http'
- Advice: Do not click any link, call any number or reply. Never share OTPs, PINs, passwords or card details. Contact the organisation only via its official app or the number printed on your card/bill.

### HIGH - likely scam  (risk 1.00)
> Tax refund of $842 is pending. Confirm your details to receive the deposit: http://tax-refund-claim.top

- Text model P(scam): **1.000**; domain risk: **0.967**
- Tactics: Authority impersonation, Reward / greed, Payment / credential request
- Evidence: tax-refund-claim.top stuffs security/finance words into the address (claim, refund); tax-refund-claim.top uses many hyphens, a common phishing pattern; '.top' is a low-cost TLD heavily used by phishing sites; domain model rates tax-refund-claim.top as phishing-like (P=0.97); manipulation tactics: Authority impersonation, Reward / greed, Payment / credential request
- Top contributing terms: contains a link, mentions money, word 'your', word 'claim', word 'http'
- Advice: Do not click any link, call any number or reply. Never share OTPs, PINs, passwords or card details. Contact the organisation only via its official app or the number printed on your card/bill.

### HIGH - likely scam  (risk 0.98)
> Your bank account is on hold. Click here to unlock: http://bit.ly/3xKLm92

- Text model P(scam): **0.954**; domain risk: **0.600**
- Tactics: Authority impersonation, Fear / threat
- Evidence: bit.ly is a link shortener that hides the real destination; manipulation tactics: Authority impersonation, Fear / threat
- Top contributing terms: contains a link, word 'your', word 'http', many digits, word 'to'
- Advice: Do not click any link, call any number or reply. Never share OTPs, PINs, passwords or card details. Contact the organisation only via its official app or the number printed on your card/bill.

### LOW - looks legitimate  (risk 0.01)
> Mom I lost my phone, this is my new number. Can you send me money for an urgent bill? I'll explain later.

- Text model P(scam): **0.005**; domain risk: **0.000**
- Tactics: Urgency / time pressure, Impersonating an acquaintance / romance
- Evidence: manipulation tactics: Urgency / time pressure, Impersonating an acquaintance / romance
- Top contributing terms: word 'new', word 'for', word 'money', word 'an', word 'this'
- Advice: No strong scam signals found. Still avoid sharing OTPs or passwords with anyone who contacts you first.

### LOW - looks legitimate  (risk 0.00)
> Hi, I'm Emma. I think I texted the wrong number, but you seem nice. Can we chat?

- Text model P(scam): **0.000**; domain risk: **0.000**
- Tactics: Impersonating an acquaintance / romance
- Evidence: manipulation tactics: Impersonating an acquaintance / romance
- Top contributing terms: word 'chat', word 'you', word 'number', ALL-CAPS text
- Advice: No strong scam signals found. Still avoid sharing OTPs or passwords with anyone who contacts you first.

### LOW - looks legitimate  (risk 0.19)
> Your OTP for login is 482913. It is valid for 10 minutes. Do not share it with anyone.

- Text model P(scam): **0.191**; domain risk: **0.000**
- Tactics: Payment / credential request
- Evidence: manipulation tactics: Payment / credential request
- Top contributing terms: many digits, word 'for', word 'your', word 'with', word 'anyone'
- Advice: No strong scam signals found. Still avoid sharing OTPs or passwords with anyone who contacts you first.

### HIGH - likely scam  (risk 0.96)
> Thanks for your payment of $45.00 to City Water. Your receipt is available in your account.

- Text model P(scam): **0.964**; domain risk: **0.000**
- Tactics: none detected
- Evidence: none
- Top contributing terms: mentions money, word 'your', word 'for your', word 'for', many digits
- Advice: Do not click any link, call any number or reply. Never share OTPs, PINs, passwords or card details. Contact the organisation only via its official app or the number printed on your card/bill.

### LOW - looks legitimate  (risk 0.32)
> Your verification code is 771204. If you did not request this, you can ignore this message.

- Text model P(scam): **0.320**; domain risk: **0.000**
- Tactics: none detected
- Evidence: none
- Top contributing terms: word 'this message', many digits, word 'your', word 'this', word 'ignore'
- Advice: No strong scam signals found. Still avoid sharing OTPs or passwords with anyone who contacts you first.

### HIGH - likely scam  (risk 1.00)
> FREE entry in 2 a weekly comp to win FA Cup tickets! Text FA to 87121 now. T&C apply 08452810075

- Text model P(scam): **1.000**; domain risk: **0.000**
- Tactics: Reward / greed
- Evidence: manipulation tactics: Reward / greed
- Top contributing terms: contains a phone number, many digits, word 'text', word 'free', word 'win'
- Advice: Do not click any link, call any number or reply. Never share OTPs, PINs, passwords or card details. Contact the organisation only via its official app or the number printed on your card/bill.

### HIGH - likely scam  (risk 1.00)
> U r a WINNER! Claim your cash prize now - call 09061701461 urgently

- Text model P(scam): **1.000**; domain risk: **0.000**
- Tactics: Reward / greed
- Evidence: manipulation tactics: Reward / greed
- Top contributing terms: contains a phone number, many digits, word 'your', word 'claim', word 'prize'
- Advice: Do not click any link, call any number or reply. Never share OTPs, PINs, passwords or card details. Contact the organisation only via its official app or the number printed on your card/bill.

### HIGH - likely scam  (risk 0.99)
> F.R.E.E entry! W1n a pr1ze - cl@1m your ca$h today

- Text model P(scam): **0.993**; domain risk: **0.000**
- Tactics: none detected
- Evidence: none
- Top contributing terms: mentions money, word 'your', many digits, word 'entry', exclamation marks
- Advice: Do not click any link, call any number or reply. Never share OTPs, PINs, passwords or card details. Contact the organisation only via its official app or the number printed on your card/bill.
