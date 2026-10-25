import os, time
os.environ['DISPLAY'] = ':0'
from Xlib import X, display, protocol

d = display.Display(':0')
root = d.screen().root

_NET_CLIENT_LIST = d.intern_atom('_NET_CLIENT_LIST')
_NET_ACTIVE_WINDOW = d.intern_atom('_NET_ACTIVE_WINDOW')
_NET_WM_NAME = d.intern_atom('_NET_WM_NAME')
WM_NAME = d.intern_atom('WM_NAME')

def get_window_name(w):
    for atom in [_NET_WM_NAME, WM_NAME]:
        prop = w.get_full_property(atom, 0)
        if prop:
            val = prop.value
            if isinstance(val, bytes):
                return val.decode('utf-8', errors='ignore')
            return str(val)
    return ''

time.sleep(2.0)
client_list = root.get_full_property(_NET_CLIENT_LIST, X.AnyPropertyType)
if client_list:
    for wid in client_list.value:
        w = d.create_resource_object('window', wid)
        name = get_window_name(w)
        if 'isaac sim' in name.lower():
            print(f'Activating Isaac Sim window: {name}', flush=True)
            event = protocol.event.ClientMessage(
                window=w,
                client_type=_NET_ACTIVE_WINDOW,
                data=(32, [1, X.CurrentTime, 0, 0, 0])
            )
            root.send_event(event, event_mask=X.SubstructureRedirectMask | X.SubstructureNotifyMask)
            w.configure(stack_mode=X.Above)
            d.sync()

time.sleep(1.5)
import mss
with mss.mss(display=':0') as sct:
    out = '/home/rum/.gemini/antigravity/brain/7b59cd11-515a-4ded-ad5e-413fbabe33fe/isaac_sim_stage9_7400_coexistence_live.png'
    sct.shot(output=out)
    print('Stage 9 screenshot captured, size:', os.path.getsize(out), flush=True)

