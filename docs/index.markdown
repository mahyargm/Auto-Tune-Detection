---
layout: default
---

---

# Abstract
<p style="text-align: justify;">In the realm of music production and audio processing, the deployment of automatic singing pitch correction, also known as Auto-Tune, has significantly transformed the landscape of vocal performance. While auto-tuning technology has offered musicians the ability to tune their vocal pitches and achieve a desired level of precision, its use has also sparked debates regarding its impact on authenticity and artistic integrity. As a result, detecting and analyzing Auto-Tuned vocals in music recordings have become essential for music scholars, producers, and listeners. However, to the best of our knowledge, no prior effort has been made in this direction. This study introduces a data-driven approach leveraging triplet networks for the detection of Auto-Tuned songs, backed by the creation of a dataset composed of original and Auto-Tuned audio clips. The experimental results demonstrate the superiority of the proposed method in both accuracy and robustness compared to Rawnet2, an end-to-end model proposed for anti-spoofing and widely used for other audio forensic tasks.</p>

**Note:** The related manuscript is currently under review.

***

# Performance on some external tracks

**Ed Sheeran - Photograph (Acoustic Version):** This song doesn't contain any type of auto-tuning and the vocals are raw and untuned.

<iframe width="100%" height="166" scrolling="no" frameborder="no" allow="autoplay" src="https://w.soundcloud.com/player/?url=https%3A//api.soundcloud.com/tracks/161084205&color=ff5500"></iframe><div style="font-size: 10px; color: #cccccc;line-break: anywhere;word-break: normal;overflow: hidden;white-space: nowrap;text-overflow: ellipsis; font-family: Interstate,Lucida Grande,Lucida Sans Unicode,Lucida Sans,Garuda,Verdana,Tahoma,sans-serif;font-weight: 100;"><a href="https://soundcloud.com/itspraddy" title="praddy" target="_blank" style="color: #cccccc; text-decoration: none;">praddy</a> · <a href="https://soundcloud.com/itspraddy/ed-sheeran-photograph-acoustic-version" title="Ed Sheeran - Photograph (Acoustic Version)" target="_blank" style="color: #cccccc; text-decoration: none;">Ed Sheeran - Photograph (Acoustic Version)</a></div>

**The Detector Predictions:** 


<div style="margin: 0  auto">
    <table align="center" style="width: 100%;">
        <tr>
            <th style="text-align:center;font-size: 15px; background-color: #D3D3D3; color: Black;">Segment Numbers</th>
            <th style="text-align:center;font-size: 15px; background-color: #D3D3D3; color: Black;">Auto-Tuned Segments (Number)</th>
            <th style="text-align:center;font-size: 15px; background-color: #D3D3D3; color: Black;">Auto-Tuned Segments (%)</th>
            <th style="text-align:center;font-size: 15px; background-color: #D3D3D3; color: Black;">Average Likelihood</th>
        </tr>
        <tr>
            <td style="text-align:center;">23</td>
            <td style="text-align:center;">0</td>
            <td style="text-align:center;">0%</td>
            <td style="text-align:center;">0.04%</td>
        </tr>
    </table>
</div>



<div style="border-bottom: 2px solid #157878; margin-top: 50px; margin-bottom: 50px;"></div>

**You & I feat. Kata Kozma:** There is low amount of pitch correction in limited segments of the song.

<iframe width="100%" height="166" scrolling="no" frameborder="no" allow="autoplay" src="https://w.soundcloud.com/player/?url=https%3A//api.soundcloud.com/tracks/892951969&color=ff5500"></iframe><div style="font-size: 10px; color: #cccccc;line-break: anywhere;word-break: normal;overflow: hidden;white-space: nowrap;text-overflow: ellipsis; font-family: Interstate,Lucida Grande,Lucida Sans Unicode,Lucida Sans,Garuda,Verdana,Tahoma,sans-serif;font-weight: 100;"><a href="https://soundcloud.com/flux-pavilion" title="Flux Pavilion" target="_blank" style="color: #cccccc; text-decoration: none;">Flux Pavilion</a> · <a href="https://soundcloud.com/flux-pavilion/flux-pavilion-you-i-feat-kata-kozma" title="You &amp; I feat. Kata Kozma" target="_blank" style="color: #cccccc; text-decoration: none;">You &amp; I feat. Kata Kozma</a></div>

**The Detector Predictions:** 


<div style="margin: 0  auto">
    <table align="center" style="width: 100%;">
        <tr>
            <th style="text-align:center;font-size: 15px; background-color: #D3D3D3; color: Black;">Segment Numbers</th>
            <th style="text-align:center;font-size: 15px; background-color: #D3D3D3; color: Black;">Auto-Tuned Segments (Number)</th>
            <th style="text-align:center;font-size: 15px; background-color: #D3D3D3; color: Black;">Auto-Tuned Segments (%)</th>
            <th style="text-align:center;font-size: 15px; background-color: #D3D3D3; color: Black;">Average Score</th>
        </tr>
        <tr>
            <td style="text-align:center;">16</td>
            <td style="text-align:center;">3</td>
            <td style="text-align:center;">18.75%</td>
            <td style="text-align:center;">18.67%</td>
        </tr>
    </table>
</div>

<div style="border-bottom: 2px solid #157878; margin-top: 50px; margin-bottom: 50px;"></div>

**Timmy Trumpet, SwedishRedElephant, 22Bullets - The City:** This song has undergone more intense auto-tuning; however it remains barely noticble for non-professional listeners.

<iframe width="100%" height="166" scrolling="no" frameborder="no" allow="autoplay" src="https://w.soundcloud.com/player/?url=https%3A//api.soundcloud.com/tracks/941750887&color=ff5500"></iframe><div style="font-size: 10px; color: #cccccc;line-break: anywhere;word-break: normal;overflow: hidden;white-space: nowrap;text-overflow: ellipsis; font-family: Interstate,Lucida Grande,Lucida Sans Unicode,Lucida Sans,Garuda,Verdana,Tahoma,sans-serif;font-weight: 100;"><a href="https://soundcloud.com/sinphonyrecs" title="SINPHONY" target="_blank" style="color: #cccccc; text-decoration: none;">SINPHONY</a> · <a href="https://soundcloud.com/sinphonyrecs/timmy-trumpet-swedishredelephant-22bullets-the-city" title="Timmy Trumpet, SwedishRedElephant, 22Bullets - The City" target="_blank" style="color: #cccccc; text-decoration: none;">Timmy Trumpet, SwedishRedElephant, 22Bullets - The City</a></div>

**The Detector Predictions:** 


<div style="margin: 0 auto">
    <table align="center" style="width: 100%">
        <tr>
            <th style="text-align:center;font-size: 15px; background-color: #D3D3D3; color: Black; border: 1px solid #white;">Segment Numbers</th>
            <th style="text-align:center;font-size: 15px; background-color: #D3D3D3; color: Black;border: 1px solid #white;">Auto-Tuned Segments (Number)</th>
            <th style="text-align:center;font-size: 15px; background-color: #D3D3D3; color: Black;border: 1px solid #white;">Auto-Tuned Segments (%)</th>
            <th style="text-align:center;font-size: 15px; background-color: #D3D3D3; color: Black; border: 1px solid #white;">Average Score</th>
        </tr>
        <tr>
            <td style="text-align:center">13</td>
            <td style="text-align:center">3</td>
            <td style="text-align:center">23.07%</td>
            <td style="text-align:center">21.95%</td>
        </tr>
    </table>
</div>


<div style="border-bottom: 2px solid #157878; margin-top: 50px; margin-bottom: 50px;"></div>

**Lua freestyle2:** In this track, there is a significant presence of intense auto-tuning throughout. The effect of auto-tuning is fully audible even to non-professional listeners.

<iframe width="100%" height="166" scrolling="no" frameborder="no" allow="autoplay" src="https://w.soundcloud.com/player/?url=https%3A//api.soundcloud.com/tracks/993188746&color=ff5500"></iframe><div style="font-size: 10px; color: #cccccc;line-break: anywhere;word-break: normal;overflow: hidden;white-space: nowrap;text-overflow: ellipsis; font-family: Interstate,Lucida Grande,Lucida Sans Unicode,Lucida Sans,Garuda,Verdana,Tahoma,sans-serif;font-weight: 100;"><a href="https://soundcloud.com/heoello" title="CookboyTheo" target="_blank" style="color: #cccccc; text-decoration: none;">CookboyTheo</a> · <a href="https://soundcloud.com/heoello/lua-freestyle-s2" title="Lua freestyle 2" target="_blank" style="color: #cccccc; text-decoration: none;">Lua freestyle 2</a></div>

**The Detector Predictions:** 


<div style="margin: 0  auto">
    <table align="center" style="width: 100%;">
        <tr>
            <th style="text-align:center;font-size: 15px; background-color: #D3D3D3; color: Black;">Segment Numbers</th>
            <th style="text-align:center;font-size: 15px; background-color: #D3D3D3; color: Black;">Auto-Tuned Segments (Number)</th>
            <th style="text-align:center;font-size: 15px; background-color: #D3D3D3; color: Black;">Auto-Tuned Segments (%)</th>
            <th style="text-align:center;font-size: 15px; background-color: #D3D3D3; color: Black;">Average Score</th>
        </tr>
        <tr>
            <td style="text-align:center;">9</td>
            <td style="text-align:center;">8</td>
            <td style="text-align:center;">88.88%</td>
            <td style="text-align:center;">88.85%</td>
        </tr>
    </table>
</div>
