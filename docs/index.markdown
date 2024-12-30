---
layout: default
---

---

# Abstract
<p style="text-align: justify;">In the domain of music production and audio processing, the implementation of automatic pitch correction of the singing voice, also known as Auto-Tune, has significantly transformed the landscape of vocal performance. While autotuning technology has offered musicians the ability to tune their vocal pitches and achieve a desired level of precision, its use has also sparked debates regarding its impact on authenticity and artistic integrity. As a result, detecting and analyzing AutoTuned vocals in music recordings has become valuable for music scholars, producers, and listeners. However, to the best of our knowledge, no prior effort has been made in this direction. This study introduces a data-driven approach leveraging triplet networks for the detection of Auto-Tuned songs, backed by the creation of a dataset composed of original and Auto-Tuned audio clips. The experimental results demonstrate the superiority of the proposed method in terms of both accuracy and robustness when compared to two baseline models: Rawnet2, an end-to-end model proposed for anti-spoofing and widely used for other audio forensic tasks, and a Graph Attention Transformer-based approach specifically designed for singing vocal deepfake detection.</p>

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
            <th style="text-align:center;font-size: 15px; background-color: #D3D3D3; color: Black;">Average Likelihood</th>
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
            <th style="text-align:center;font-size: 15px; background-color: #D3D3D3; color: Black; border: 1px solid #white;">Average Likelihood</th>
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
            <th style="text-align:center;font-size: 15px; background-color: #D3D3D3; color: Black;">Average Likelihood</th>
        </tr>
        <tr>
            <td style="text-align:center;">9</td>
            <td style="text-align:center;">8</td>
            <td style="text-align:center;">88.88%</td>
            <td style="text-align:center;">88.85%</td>
        </tr>
    </table>
</div>

***

# Runtime evaluation

The average runtime performance of the proposed method on the test dataset (10-second segmetns), evaluated across various backbone architectures, is summarized in the table below. The experiments were conducted on a system with the following specifications:
- **CPU:** AMD EPYC 7742 64-Core Processor
- **RAM:** 32 GB
- **GPU:** NVIDIA A100
- **Operating System:** Ubuntu 22.04.4 LTS

<div style="margin: 0 auto; width: 100%;">
    <table align="center" style="width: 100%;">
        <tr>
            <th style="text-align:center; font-size: 15px; background-color: #D3D3D3; color: Black; border: 1px solid #ccc; width: 25%;">Backbones</th>
            <th style="text-align:center; font-size: 15px; background-color: #D3D3D3; color: Black; border: 1px solid #ccc; width: 25%;">Feature Extractor (ms)</th>
            <th style="text-align:center; font-size: 15px; background-color: #D3D3D3; color: Black; border: 1px solid #ccc; width: 25%;">Classifier (ms)</th>
            <th style="text-align:center; font-size: 15px; background-color: #D3D3D3; color: Black; border: 1px solid #ccc; width: 25%;">Total (ms)</th>
        </tr>
        <tr>
            <td style="text-align:center; border: 1px solid #ccc; padding: 10px;">ResNeXt</td>
            <td style="text-align:center; border: 1px solid #ccc; padding: 10px;">10.562</td>
            <td style="text-align:center; border: 1px solid #ccc; padding: 10px;">0.285</td>
            <td style="text-align:center; border: 1px solid #ccc; padding: 10px;">10.847</td>
        </tr>
        <tr>
            <td style="text-align:center; border: 1px solid #ccc; padding: 10px;">EfficientNet</td>
            <td style="text-align:center; border: 1px solid #ccc; padding: 10px;">21.350</td>
            <td style="text-align:center; border: 1px solid #ccc; padding: 10px;">0.285</td>
            <td style="text-align:center; border: 1px solid #ccc; padding: 10px;">21.635</td>
        </tr>
        <tr>
            <td style="text-align:center; border: 1px solid #ccc; padding: 10px;">ResNet18</td>
            <td style="text-align:center; border: 1px solid #ccc; padding: 10px;">5.741</td>
            <td style="text-align:center; border: 1px solid #ccc; padding: 10px;">0.285</td>
            <td style="text-align:center; border: 1px solid #ccc; padding: 10px;"><strong>6.026</strong></td>
        </tr>
        <tr>
            <td style="text-align:center; border: 1px solid #ccc; padding: 10px;">ResNet50</td>
            <td style="text-align:center; border: 1px solid #ccc; padding: 10px;">10.457</td>
            <td style="text-align:center; border: 1px solid #ccc; padding: 10px;">0.285</td>
            <td style="text-align:center; border: 1px solid #ccc; padding: 10px;">10.742</td>
        </tr>
        <tr>
            <td style="text-align:center; border: 1px solid #ccc; padding: 10px;">B01</td>
            <td style="text-align:center; border: 1px solid #ccc; padding: 10px;">-</td>
            <td style="text-align:center; border: 1px solid #ccc; padding: 10px;">-</td>
            <td style="text-align:center; border: 1px solid #ccc; padding: 10px;">13.379</td>
        </tr>
        <tr>
            <td style="text-align:center; border: 1px solid #ccc; padding: 10px;">B02</td>
            <td style="text-align:center; border: 1px solid #ccc; padding: 10px;">-</td>
            <td style="text-align:center; border: 1px solid #ccc; padding: 10px;">-</td>
            <td style="text-align:center; border: 1px solid #ccc; padding: 10px;">10.601</td>
        </tr>
    </table>
</div>


