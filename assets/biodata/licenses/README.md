# Biodata PDF assets (biodata_pdf.py)

Copied from the app repo (MauiApp2), the same files the app ships. Nothing here was downloaded.

| File | From (MauiApp2 repo) | Licence |
|---|---|---|
| `fonts/PlusJakartaSans-Regular.ttf`, `-SemiBold.ttf`, `-Bold.ttf` | `MauiApp2/Resources/Fonts` | SIL OFL 1.1, `PlusJakartaSans-OFL.txt` |
| `fonts/BricolageGrotesque-Bold.ttf` | `MauiApp2/Resources/Fonts` | SIL OFL 1.1, `BricolageGrotesque-OFL.txt` |
| `fonts/Lucide.ttf` (icons) | `MauiApp2/Resources/Fonts` | ISC, `Lucide-ISC.txt` |
| `appicon_128.png` | `MauiApp2/Resources/AppIcon/appicon.png`, resized to 128 px | Shaadi Parrot's own logo |
| `mithu_256.png` | `MauiApp2/Resources/Images/bioparrot.webp`, as a 256 px PNG | Shaadi Parrot's own art |

Modifications (as in the app, see MauiApp2 `docs/licenses/README.md`): static instances cut from the variable
fonts, the Plus Jakarta Sans space widened, vertical metrics aligned, and Lucide subset to the icons the app uses.
The PDF uses seven Lucide glyphs: UserRound U+E468, ScrollText U+E45F, GraduationCap U+E234, Users U+E1A4,
MoonStar U+E410, Phone U+E133, BadgeCheck U+E241. fpdf2 embeds only the glyphs a PDF uses.
