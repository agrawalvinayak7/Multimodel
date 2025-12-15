# UI/UX Upgrade Prompt for Sentiment Analysis SaaS Application

## Project Overview
You are tasked with completely redesigning the UI/UX of a video sentiment analysis SaaS application. The app analyzes videos to detect emotions and sentiments. The current codebase uses Next.js 15, React 19, Tailwind CSS 4, TypeScript, and Prisma.

## Theme Direction: Inside Out (Pixar) Inspired
Create a full Inside Out themed interface that captures the essence of the Pixar movie - emotions as central characters, vibrant colors, and an immersive emotional headquarters aesthetic.

---

## Design Philosophy

### Core Principles
1. **Emotion-Forward Design**: Every UI element should celebrate emotions and make sentiment analysis feel magical and intuitive
2. **Glass Morphism**: Use frosted glass effects (backdrop-blur, semi-transparent backgrounds) for all major UI components
3. **Immersive Backgrounds**: Each page should have a full-screen, cinematic background image that tells a story
4. **Smooth Interactions**: All transitions, hover effects, and animations should be butter-smooth (use CSS transitions with ease-in-out)
5. **Clean & Uncluttered**: Despite being colorful, maintain generous white space and clear visual hierarchy
6. **Professional Yet Playful**: Balance whimsy with usability - this is a B2B SaaS product that happens to be delightful

---

## Color Palette (Inside Out Inspired)

### Primary Emotion Colors
- **Joy (Yellow)**: `#FFD700`, `#FFA500`, `#FFEB3B` - Use for positive sentiments, success states, primary CTAs
- **Sadness (Blue)**: `#4A90E2`, `#2196F3`, `#5B9BD5` - Use for neutral states, information
- **Anger (Red)**: `#E74C3C`, `#F44336`, `#D32F2F` - Use for negative sentiments, errors, warnings
- **Fear (Purple)**: `#9B59B6`, `#7B1FA2`, `#8E44AD` - Use for alerts, cautionary messages
- **Disgust (Green)**: `#27AE60`, `#4CAF50`, `#2ECC71` - Use for secondary actions
- **Neutral (Beige/Cream)**: `#F5F5DC`, `#FAF3E0`, `#FFF8E7` - Use for backgrounds, cards

### Supporting Colors
- **Dark Mode Base**: `#1A1A2E`, `#16213E`, `#0F3460` - Deep blues/purples for dark backgrounds
- **Glass Overlay**: `rgba(255, 255, 255, 0.1)` to `rgba(255, 255, 255, 0.25)` with backdrop-blur
- **Gradient Overlays**: Multi-color gradients combining 2-3 emotion colors

### Text Colors
- **Primary Text**: `#2C3E50` (dark mode: `#ECEFF1`)
- **Secondary Text**: `#7F8C8D` (dark mode: `#B0BEC5`)
- **Muted Text**: `#95A5A6` (dark mode: `#90A4AE`)

---

## Typography

### Font Families
- **Headings**: Use `'Poppins'`, `'Quicksand'`, or `'Fredoka'` (rounded, friendly fonts)
- **Body**: Use `'Inter'`, `'DM Sans'`, or keep `Geist` (clean, readable)
- **Monospace** (for API keys): `'Fira Code'`, `'JetBrains Mono'`

### Font Weights & Sizes
- **Hero Headings**: 2.5rem - 3rem, font-weight: 700-800
- **Section Headings**: 1.5rem - 2rem, font-weight: 600-700
- **Body Text**: 0.875rem - 1rem, font-weight: 400-500
- **Small Text**: 0.75rem - 0.875rem, font-weight: 400

---

## Component Design System

### Glass Morphism Components
All major UI components should use this glass effect:

```css
.glass-card {
  background: rgba(255, 255, 255, 0.15);
  backdrop-filter: blur(20px);
  -webkit-backdrop-filter: blur(20px);
  border: 1px solid rgba(255, 255, 255, 0.2);
  border-radius: 16px;
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
}
```

### Buttons

#### Primary Button (Joy/CTA)
- Background: Gradient from `#FFD700` to `#FFA500`
- Hover: Scale(1.02), brightness increase, subtle glow effect
- Active: Scale(0.98)
- Padding: `px-6 py-3` (larger buttons for important actions)
- Border-radius: `12px`
- Font-weight: 600
- Transition: `all 0.3s ease-in-out`
- Add subtle box-shadow with emotion color

#### Secondary Button
- Glass morphism background
- Border: 2px solid with emotion color
- Hover: Fill with emotion color, text turns white
- Transition: `all 0.3s ease-in-out`

#### Icon Buttons
- Circular or rounded square
- Glass background
- Hover: Slight rotation (2-3deg), scale(1.1)
- Active state: Pulse animation

### Input Fields

#### Standard Input
- Background: `rgba(255, 255, 255, 0.2)` with backdrop-blur
- Border: 1px solid `rgba(255, 255, 255, 0.3)`
- Focus: Border changes to emotion color (Joy yellow), subtle glow
- Padding: `px-4 py-3`
- Border-radius: `10px`
- Placeholder: Slightly playful text, low opacity
- Transition: `all 0.2s ease`

#### File Upload Area
- Large dashed border with glass background
- Icon: Animated on hover (gentle bounce or float)
- Drag-over state: Border becomes solid, background slightly more opaque, scale(1.01)
- Upload progress: Colorful gradient progress bar with emotion colors

### Cards

#### Information Cards (API Key, Quota)
- Glass morphism background
- Subtle gradient border (use 2 emotion colors)
- Hover: Slight lift effect (translateY(-4px)), increased shadow
- Inner content: Well-spaced, clear typography
- Icons: Emotion-colored, medium size

#### Analysis Result Cards
- Glass background
- Each utterance card should have:
  - Time range badge (rounded, emotion-colored)
  - Text snippet (truncated if long)
  - Emotion bars: Horizontal progress bars with emotion colors and emojis
  - Sentiment bars: Similar style
  - Smooth reveal animation when loaded

### Navigation Bar

#### Design
- Fixed top, glass morphism background
- Height: `4rem` (64px)
- Backdrop-blur: Strong (20px+)
- Logo section: 
  - Icon + Text side by side
  - Logo should be vibrant, emotion-themed
- Right section: User menu, logout button
- Border-bottom: Subtle, 1px with low opacity
- Sticky behavior: Smooth transition when scrolling

#### Logo Redesign Brief
**USE NANO BANAN PRO TO GENERATE THE LOGO**

Create a logo that combines:
- Abstract representation of emotions/sentiments (could be orbiting circles, emotion bubbles, or abstract faces)
- Modern, geometric style
- Incorporate 2-3 emotion colors (Joy yellow, Sadness blue, maybe a third)
- Should work in both icon-only and icon+text formats
- Icon should be recognizable at small sizes
- Style: Somewhere between abstract and representational - think "emotion headquarters control console" or "emotional spectrum"
- Export as SVG for scalability
- Provide both light and dark mode variants if needed

---

## Page-by-Page Specifications

### 1. LOGIN PAGE (`/login`)

#### Layout Structure
- **Full-screen split layout** (similar to reference image)
- **Left Side (40% width)**:
  - Glass morphism card containing login form
  - Vertically and horizontally centered
  - Max-width: 450px
  - Padding: 48px
  - Drop shadow: Strong but soft
  
- **Right Side (60% width)**:
  - Full-height background image
  - Should depict "Inside Out headquarters" aesthetic or emotional/colorful abstract scene
  - Subtle parallax effect on mouse move (optional enhancement)

#### Background Image Specifications
- **Style**: Cinematic, vibrant, emotion-themed
- **Suggestions**:
  - Abstract representation of emotions as glowing orbs/spheres
  - Colorful gradient landscape (sunset/aurora style with emotion colors)
  - Stylized "control center" with screens and emotion data
  - Memory orbs floating in space (Inside Out reference)
- **Requirements**:
  - High resolution (1920x1080 minimum)
  - Warm color palette with pops of emotion colors
  - Should NOT distract from the form
  - Subtle grain or texture overlay for depth

#### Login Form Components

**Header Section**:
- Welcome text: "Welcome Back!" (font-size: 2rem, weight: 700)
- Subtitle: "Sign in to analyze emotions" (font-size: 0.875rem, text-gray-600)
- Both centered above form

**Form Elements**:
1. Email Input:
   - Label: "Email address" (small, medium weight)
   - Placeholder: "joy@insideout.com" or "your@email.com"
   - Type: email
   - Validation: Show inline error with red accent if invalid

2. Password Input:
   - Label: "Password"
   - Placeholder: "••••••••"
   - Type: password with toggle visibility icon
   - Toggle icon: Eye/eye-off, emotion-colored on hover

3. Error Message:
   - If error exists: Show in red glass card above form
   - Icon + text
   - Smooth slide-down animation

4. Submit Button:
   - Full width
   - Text: "Log in" or "Sign In"
   - Loading state: Text changes to "Logging in..." with spinner
   - Disabled state: Reduced opacity, no hover effects

5. Footer Link:
   - Text: "Don't have an account? Sign up"
   - "Sign up" should be emotion-colored (Joy yellow) and underlined on hover

#### Animations & Interactions

**Page Load**:
- Form fades in from left (opacity 0 → 1, translateX(-20px → 0))
- Background image fades in from right
- Duration: 0.6s, ease-out
- Stagger: Form elements appear one by one (0.1s delay each)

**Input Focus**:
- Border glows with Joy yellow
- Subtle scale(1.01)
- Placeholder slides up slightly

**Button Hover**:
- Scale(1.02)
- Gradient shifts/animates
- Box-shadow intensifies (glow effect)

**Button Click**:
- Scale(0.98) momentarily
- Ripple effect from click point

**Error State**:
- Shake animation on form
- Error card slides down with bounce

---

### 2. SIGNUP PAGE (`/signup`)

#### Layout
- **Same layout as login** (split screen with glass form card)
- **Background**: Different but complementary image to login
  - Suggestion: More "new beginning" themed - bright, optimistic, Joy-forward

#### Form Components
- All inputs follow same design as login
- Additional fields:
  1. Full Name input (before email)
  2. Confirm Password input (after password)
- Password strength indicator:
  - Progress bar below password field
  - Colors: Red (weak) → Yellow (medium) → Green (strong)
  - Text indicator: "Weak", "Medium", "Strong"

#### Additional Elements
- Checkbox for terms acceptance:
  - Custom styled checkbox (rounded, emotion-colored when checked)
  - Label with link to terms (link should be blue, underlined on hover)
- Footer link: "Already have an account? Sign in"

#### Animations
- Same as login page
- Password strength bar animates as user types (width transition)

---

### 3. MAIN DASHBOARD / HOMEPAGE (`/`)

#### Background
- **Full viewport height background**
- **Style**: "Inside Out headquarters" control room aesthetic
  - Multiple emotion-colored screens/monitors in background
  - Subtle animated elements (floating particles, gentle pulsing glows)
  - Dark base with vibrant accent lights
- **Implementation**: 
  - Fixed background image or CSS gradient with overlays
  - Add `::before` pseudo-element with subtle animated gradient overlay

#### Layout Structure
- **Navigation**: Glass nav bar at top (as described earlier)
- **Main Content**: Two-column layout (stacks on mobile)
  - **Left Column (50%)**:
    - Inference section (video upload + results)
  - **Right Column (50%)**:
    - API information section
    - Code examples section

#### Navigation Bar
- Logo + "Sentiment Analysis" text (use new logo from Nano Banan Pro)
- Right side: Logout button (glass button with icon)
- Hover effects on all interactive elements

#### Left Column: Inference Section

**Section Header**:
- "Inference" heading (font-size: 1.5rem, weight: 600)
- Emotion-colored accent line below (2px, gradient)

**Upload Component**:
- Large glass card with dashed border
- Icon: Upload cloud icon (emotion-colored, animated on hover)
- Heading: "Upload a video" (medium size)
- Subtext: "Get started with sentiment detection by uploading a video under one minute long."
- **Drag & Drop States**:
  - Default: Subtle pulse animation on icon
  - Hover: Border becomes solid, slight scale up
  - Drag-over: Background becomes more opaque, border color changes to Joy yellow
  - Uploading: Progress bar appears, icon changes to spinning loader
  - Analyzing: Progress bar + "Analyzing..." text with animated dots

**Overall Analysis Card**:
- Glass card, two-column grid inside
- Left: "Primary emotion" with large emoji (4rem size) and confidence score
- Right: "Primary Sentiment" with large emoji and confidence score
- Empty state: Dashed border with message "Upload a video to see overall analysis"
- Reveal animation when data loads: Fade in + scale from 0.95 to 1

**Utterances Analysis Section**:
- Section header: "Analysis of utterances"
- Each utterance card:
  - Glass background
  - Three sections: Time + Text | Emotions | Sentiments
  - **Time Badge**: Rounded pill with time range, emotion-colored background
  - **Text**: Truncated with "..." if too long, light gray color
  - **Emotion Bars**:
    - Label with emoji + emotion name
    - Horizontal progress bar with emotion color
    - Percentage text at end
    - Smooth fill animation (0 → actual percentage, 0.5s delay stagger)
  - **Sentiment Bars**: Same style as emotion bars
  - Cards appear one by one with stagger animation (0.1s delay each)

#### Right Column: API Section

**Secret Key Card**:
- Glass card
- Header: "Secret key" (medium weight)
- Description text: Explain purpose, smaller font, gray
- Key display:
  - Label "Key" on left
  - Key value in monospace font, truncated if needed, scrollable
  - Copy button on right:
    - Icon: Copy icon (changes to checkmark when copied)
    - Glass background
    - Hover: Slight scale, rotation
    - Copied state: Green background, checkmark icon, "Copied!" text

**Monthly Quota Card**:
- Glass card
- Header: "Monthly quota"
- Usage text: "X / Y requests" (right-aligned)
- Progress bar:
  - Background: Light gray
  - Fill: Emotion-colored gradient (changes color based on usage %)
    - 0-50%: Green (good)
    - 50-80%: Yellow (warning)
    - 80-100%: Red (critical)
  - Smooth transition when value changes
  - Height: 4-6px, rounded

**Code Examples Card**:
- Glass card
- Header: "API Usage"
- Description: "Examples of how to use the API..."
- Tab navigation:
  - Two tabs: "TypeScript" and "cURL"
  - Active tab: Emotion-colored bottom border, white text
  - Inactive tab: Gray text, hover effect
  - Smooth underline transition between tabs
- Code block:
  - Dark background (matching Inside Out dark aesthetic)
  - Syntax highlighting with emotion colors:
    - Strings: Joy yellow
    - Keywords: Sadness blue
    - Functions: Fear purple
    - Comments: Muted gray
  - Scrollable if content overflows
  - Copy button in top-right corner

---

## Responsive Design Specifications

### Breakpoints
- **Mobile**: < 640px
- **Tablet**: 640px - 1024px
- **Desktop**: > 1024px

### Mobile Adaptations

#### Login/Signup Pages
- Switch to single-column layout
- Form card takes full width (with padding)
- Background image becomes full-page background behind form
- Form: Max-width 400px, centered, with margin top/bottom

#### Dashboard
- Stack columns vertically
- Left column (inference) appears first
- Right column (API) appears below
- Each card full width with margin-bottom
- Navigation: Hamburger menu for any additional items (if added later)
- Text sizes slightly reduced but still readable
- Touch targets: Minimum 44x44px for all interactive elements

### Tablet Adaptations
- Similar to desktop but with smaller gaps
- Cards may be slightly narrower
- Font sizes scaled down proportionally

### Interaction Optimizations
- All hover effects also trigger on touch devices as active states
- Increased tap target sizes on mobile
- Swipe gestures for tab navigation on mobile (optional enhancement)

---

## Animation & Transition Library

### Standard Transitions
```css
transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
```

### Entrance Animations
- **Fade In Up**: opacity 0→1, translateY(20px→0), 0.4s ease-out
- **Fade In Left**: opacity 0→1, translateX(-20px→0), 0.4s ease-out
- **Scale In**: opacity 0→1, scale(0.95→1), 0.3s ease-out
- **Slide Down**: opacity 0→1, translateY(-10px→0), 0.3s ease-out

### Hover Animations
- **Lift**: translateY(-4px) + shadow increase, 0.2s
- **Grow**: scale(1.02-1.05), 0.2s
- **Glow**: box-shadow with emotion color, 0.3s
- **Rotate**: rotate(2-3deg), 0.2s

### Loading States
- **Spinner**: Circular, emotion-colored, 1s linear infinite rotation
- **Pulse**: opacity 0.5→1→0.5, 1.5s ease-in-out infinite
- **Progress Bar**: Width 0→100%, smooth transition, gradient background

### Micro-interactions
- **Button Click**: Scale down (0.98) then up (1.02) then normal (1), total 0.3s
- **Input Focus**: Border glow with emotion color, 0.2s
- **Checkbox Check**: Checkmark draws in (stroke-dashoffset animation), 0.3s
- **Toast Notifications**: Slide in from top, auto-dismiss after 3s, slide out

---

## Accessibility Requirements

### MUST HAVES
1. **Color Contrast**: All text must have WCAG AA contrast ratios (4.5:1 for normal text, 3:1 for large text)
   - Test emotion colors against backgrounds, adjust opacity if needed
2. **Focus States**: All interactive elements must have visible focus rings (2px solid emotion color)
3. **Keyboard Navigation**: All functionality accessible via keyboard
4. **ARIA Labels**: Proper labels for all form inputs, buttons, and interactive elements
5. **Alt Text**: Descriptive alt text for all images (background images can use aria-hidden)
6. **Screen Reader**: Ensure screen readers can navigate and understand all content

### NICE TO HAVES
- Reduced motion preferences: Disable animations for users who prefer reduced motion
- High contrast mode support
- Keyboard shortcuts for common actions

---

## Implementation Guidelines

### DO's
✅ Use Tailwind CSS utility classes exclusively (already installed)
✅ Use CSS custom properties for emotion colors (define in globals.css)
✅ Keep all logic intact - ONLY modify JSX and styling
✅ Use Next.js Image component for any images added
✅ Keep existing component structure, just enhance styling
✅ Add smooth transitions to all state changes
✅ Test on both light and dark mode (if applicable)
✅ Use semantic HTML (header, main, section, article, etc.)
✅ Add proper TypeScript types for any new props
✅ Keep existing data fetching and API logic untouched

### DON'Ts
❌ Do NOT modify any server actions or API routes
❌ Do NOT change authentication logic
❌ Do NOT alter database queries or Prisma schema
❌ Do NOT change the core functionality of components
❌ Do NOT use inline styles (use Tailwind classes)
❌ Do NOT add new dependencies unless absolutely necessary
❌ Do NOT modify environment variables or configuration files
❌ Do NOT change routing structure

---

## Image Asset Requirements

### Background Images Needed

1. **Login Page Background**:
   - Dimensions: 1920x1080 (landscape)
   - Style: Inside Out inspired, emotional, warm tones
   - Vibe: Welcoming, returning home
   - Suggested: Emotional headquarters with glowing screens, or abstract emotion orbs in space

2. **Signup Page Background**:
   - Dimensions: 1920x1080 (landscape)
   - Style: Inside Out inspired, bright, optimistic
   - Vibe: New beginnings, excitement
   - Suggested: Bright emotion colors radiating outward, or new memory formation scene

3. **Dashboard Background**:
   - Dimensions: 2560x1440 (or larger for scrolling)
   - Style: Inside Out headquarters control room
   - Vibe: Professional but magical, high-tech emotional analysis center
   - Suggested: Multiple emotion-colored monitors/screens, holographic emotion data, particle effects
   - Should be darker to allow glass cards to pop

### Logo Asset

**Generate using Nano Banan Pro**:
- Icon + Text version: SVG, 200x50px
- Icon only: SVG, 50x50px
- Style: Modern, geometric, emotion-themed
- Colors: Incorporate Joy yellow and Sadness blue primarily
- Export: SVG format for scalability
- Provide transparent background version

### Additional Icons/Graphics (Optional)
- Emotion emojis: Can use Unicode emojis or custom designed emotion icons
- Upload icon: Cloud with upward arrow
- Copy icon, eye icon, etc.: Use from react-icons library (already installed)

---

## Code Organization

### File Structure to Modify
```
frontend/src/
├── app/
│   ├── login/page.tsx          ← Modify: Update JSX structure and Tailwind classes
│   ├── signup/page.tsx         ← Modify: Update JSX structure and Tailwind classes
│   └── page.tsx                ← Modify: Update JSX structure and Tailwind classes
├── components/client/
│   ├── CodeExamples.tsx        ← Modify: Update styling
│   ├── copy-button.tsx         ← Modify: Update styling and animations
│   ├── inference.tsx           ← Modify: Update styling, keep logic
│   ├── Signout.tsx             ← Modify: Update styling
│   └── UploadVideo.tsx         ← Modify: Update styling, keep upload logic
├── styles/
│   └── globals.css             ← Modify: Add custom CSS variables for emotion colors
└── app/layout.tsx              ← Possibly modify: Add emotion color CSS variables
```

### CSS Variables to Add (in globals.css)

```css
:root {
  /* Emotion Colors */
  --emotion-joy: #FFD700;
  --emotion-sadness: #4A90E2;
  --emotion-anger: #E74C3C;
  --emotion-fear: #9B59B6;
  --emotion-disgust: #27AE60;
  
  /* Glass Morphism */
  --glass-bg: rgba(255, 255, 255, 0.15);
  --glass-border: rgba(255, 255, 255, 0.2);
  
  /* Shadows */
  --shadow-glass: 0 8px 32px rgba(0, 0, 0, 0.1);
  --shadow-hover: 0 12px 40px rgba(0, 0, 0, 0.15);
}
```

---

## Testing Checklist

After implementing, verify:

- [ ] All pages load without console errors
- [ ] Login/logout flow works correctly
- [ ] Video upload and analysis functionality intact
- [ ] API key copying works
- [ ] Form validation still works
- [ ] Responsive design works on mobile (375px width)
- [ ] Responsive design works on tablet (768px width)
- [ ] Responsive design works on desktop (1920px width)
- [ ] All hover effects work smoothly
- [ ] All animations are smooth (60fps)
- [ ] Color contrast meets WCAG AA standards
- [ ] Keyboard navigation works throughout
- [ ] Focus states are visible
- [ ] Loading states display correctly
- [ ] Error states display correctly
- [ ] Glass morphism effects render correctly
- [ ] Background images load and display properly

---

## Priority Order

Implement in this order:

1. **Phase 1: Foundation**
   - Add CSS variables for emotion colors
   - Create background images or gradients for each page
   - Update globals.css with base glass morphism styles

2. **Phase 2: Login/Signup**
   - Redesign login page with glass card and background
   - Add animations and transitions
   - Redesign signup page (similar to login)
   - Test authentication flow

3. **Phase 3: Dashboard Layout**
   - Update navigation bar
   - Implement dashboard background
   - Create glass card structure for main content

4. **Phase 4: Dashboard Components**
   - Update upload component
   - Update analysis display components
   - Update API key and quota cards
   - Update code examples component

5. **Phase 5: Polish**
   - Add logo (generated from Nano Banan Pro)
   - Fine-tune animations
   - Optimize responsive behavior
   - Add micro-interactions
   - Test thoroughly

6. **Phase 6: Accessibility**
   - Test keyboard navigation
   - Verify color contrast
   - Add ARIA labels where needed
   - Test with screen reader (if possible)

---

## Final Notes

- **Maintain Functionality**: The core goal is UI/UX enhancement. All existing features must continue to work.
- **Performance**: Ensure animations don't cause jank. Use `transform` and `opacity` for animations (GPU accelerated).
- **Browser Support**: Test on Chrome, Firefox, Safari (Webkit for backdrop-filter support).
- **Consistency**: Use the established design system throughout. Every button, input, card should follow the same patterns.
- **Emotion-Forward**: At every decision point, ask "does this celebrate emotions?" The UI should feel alive and emotionally intelligent.

---

## Success Criteria

The redesign is successful when:
1. ✅ The app feels like it belongs in the Inside Out universe
2. ✅ Glass morphism is implemented consistently and beautifully
3. ✅ All animations are smooth and delightful
4. ✅ The UI is fully responsive on all device sizes
5. ✅ Core functionality (auth, upload, analysis, API) works perfectly
6. ✅ The design is accessible (WCAG AA compliance)
7. ✅ Users feel excited and engaged when using the app
8. ✅ The brand identity is strong and memorable (new logo + cohesive theme)

---

## Questions or Clarifications?

If you need clarification on any aspect:
1. Refer to Inside Out movie screenshots for inspiration
2. Use glass morphism examples from Dribbble or Behance
3. Test emotion color combinations for optimal contrast
4. When in doubt, prioritize user experience over visual complexity

**Good luck, and create something emotionally magical! 🎨✨**