import json, wave
from pathlib import Path
import numpy as np
rows=[]
for emotion in ('neutral','excited','somber'):
 p=Path('artifacts/explicit_neural')/(emotion+'.wav')
 with wave.open(str(p)) as w:
  sr=w.getframerate(); y=np.frombuffer(w.readframes(w.getnframes()),dtype='<i2').astype(float)/32768
 pitches=[]
 for start in range(0,len(y)-1200,300):
  z=y[start:start+1200]; z=z-z.mean()
  if np.sqrt(np.mean(z*z))<.02: continue
  c=np.correlate(z,z,mode='full')[1199:]
  lags=np.arange(60,344)
  normalized=np.array([c[k]/np.sqrt(np.dot(z[:-k],z[:-k])*np.dot(z[k:],z[k:])) for k in lags])
  peaks=np.flatnonzero((normalized[1:-1]>normalized[:-2])&(normalized[1:-1]>=normalized[2:]))+1
  if not len(peaks): continue
  best=normalized[peaks].max()
  if best<.7: continue
  chosen=peaks[normalized[peaks]>=max(.7,best*.95)][0]
  pitches.append(sr/lags[chosen])
 row={'emotion':emotion,'median_voiced_hz':float(np.median(pitches)), 'voiced_frames':len(pitches), 'rms':float(np.sqrt(np.mean(y*y))), 'seconds':len(y)/sr}
 rows.append(row);print(row,flush=True)
assert rows[1]['median_voiced_hz']>rows[0]['median_voiced_hz']>rows[2]['median_voiced_hz']
Path('artifacts/explicit_neural/tone_measurements.json').write_text(json.dumps({'method':'normalized waveform autocorrelation; 1200 window, 300 hop, 70..400 Hz, RMS>.02, confidence>.7; approximate voiced medians','results':rows},indent=2)+'\n')
