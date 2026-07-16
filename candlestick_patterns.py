import pandas as pd
import pandas_ta as ta

def detect_candlestick_patterns(df):
    patterns = {}
    if len(df) < 3:
        return patterns

    try:
        cdl = ta.cdl_pattern(df['open'], df['high'], df['low'], df['close'], name='all')
        if cdl is not None:
            last = cdl.iloc[-1]
            for col in last.index:
                val = last[col]
                if val == 100:
                    patterns[col] = 'bullish'
                elif val == -100:
                    patterns[col] = 'bearish'
    except Exception as e:
        print(f"⚠️ Pandas TA pattern detection error: {e}")

    try:
        open_vals = df['open'].values[-3:]
        high_vals = df['high'].values[-3:]
        low_vals = df['low'].values[-3:]
        close_vals = df['close'].values[-3:]

        c1 = {'open': open_vals[0], 'high': high_vals[0], 'low': low_vals[0], 'close': close_vals[0]}
        c2 = {'open': open_vals[1], 'high': high_vals[1], 'low': low_vals[1], 'close': close_vals[1]}
        c3 = {'open': open_vals[2], 'high': high_vals[2], 'low': low_vals[2], 'close': close_vals[2]}

        def body_color(open_, close_):
            return 'green' if close_ > open_ else 'red'

        def body_size(open_, close_):
            return abs(close_ - open_)

        # Morning Star
        if (body_color(c1['open'], c1['close']) == 'red' and
            body_size(c1['open'], c1['close']) > body_size(c2['open'], c2['close']) * 0.5 and
            body_size(c2['open'], c2['close']) < body_size(c1['open'], c1['close']) * 0.3 and
            body_color(c3['open'], c3['close']) == 'green' and
            c3['close'] > c1['close']):
            patterns['MORNING_STAR'] = 'bullish'

        # Evening Star
        if (body_color(c1['open'], c1['close']) == 'green' and
            body_size(c1['open'], c1['close']) > body_size(c2['open'], c2['close']) * 0.5 and
            body_size(c2['open'], c2['close']) < body_size(c1['open'], c1['close']) * 0.3 and
            body_color(c3['open'], c3['close']) == 'red' and
            c3['close'] < c1['close']):
            patterns['EVENING_STAR'] = 'bearish'

        # Morning Doji Star
        if (body_color(c1['open'], c1['close']) == 'red' and
            abs(c2['open'] - c2['close']) < 0.0001 and
            body_color(c3['open'], c3['close']) == 'green' and
            c3['close'] > c1['close']):
            patterns['MORNING_DOJI_STAR'] = 'bullish'

        # Evening Doji Star
        if (body_color(c1['open'], c1['close']) == 'green' and
            abs(c2['open'] - c2['close']) < 0.0001 and
            body_color(c3['open'], c3['close']) == 'red' and
            c3['close'] < c1['close']):
            patterns['EVENING_DOJI_STAR'] = 'bearish'

        # Three White Soldiers
        if (body_color(c1['open'], c1['close']) == 'green' and
            body_color(c2['open'], c2['close']) == 'green' and
            body_color(c3['open'], c3['close']) == 'green' and
            c1['close'] < c2['close'] < c3['close'] and
            body_size(c1['open'], c1['close']) > 0.001 and
            body_size(c2['open'], c2['close']) > 0.001 and
            body_size(c3['open'], c3['close']) > 0.001):
            patterns['THREE_WHITE_SOLDIERS'] = 'bullish'

        # Three Black Crows
        if (body_color(c1['open'], c1['close']) == 'red' and
            body_color(c2['open'], c2['close']) == 'red' and
            body_color(c3['open'], c3['close']) == 'red' and
            c1['close'] > c2['close'] > c3['close'] and
            body_size(c1['open'], c1['close']) > 0.001 and
            body_size(c2['open'], c2['close']) > 0.001 and
            body_size(c3['open'], c3['close']) > 0.001):
            patterns['THREE_BLACK_CROWS'] = 'bearish'

        # Three Inside Up
        if (body_color(c1['open'], c1['close']) == 'red' and
            c2['open'] > c1['close'] and c2['close'] < c1['open'] and
            body_color(c3['open'], c3['close']) == 'green' and
            c3['close'] > c2['close']):
            patterns['THREE_INSIDE_UP'] = 'bullish'

        # Three Inside Down
        if (body_color(c1['open'], c1['close']) == 'green' and
            c2['open'] < c1['close'] and c2['close'] > c1['open'] and
            body_color(c3['open'], c3['close']) == 'red' and
            c3['close'] < c2['close']):
            patterns['THREE_INSIDE_DOWN'] = 'bearish'

        # Three Outside Up (Bullish Engulfing + confirmation)
        if (body_color(c1['open'], c1['close']) == 'red' and
            c2['open'] < c1['close'] and c2['close'] > c1['open'] and
            body_color(c3['open'], c3['close']) == 'green' and
            c3['close'] > c2['close']):
            patterns['THREE_OUTSIDE_UP'] = 'bullish'

        # Three Outside Down (Bearish Engulfing + confirmation)
        if (body_color(c1['open'], c1['close']) == 'green' and
            c2['open'] > c1['close'] and c2['close'] < c1['open'] and
            body_color(c3['open'], c3['close']) == 'red' and
            c3['close'] < c2['close']):
            patterns['THREE_OUTSIDE_DOWN'] = 'bearish'

        # Piercing Line
        if (body_color(c1['open'], c1['close']) == 'red' and
            body_color(c2['open'], c2['close']) == 'green' and
            c2['open'] < c1['close'] and
            c2['close'] > (c1['open'] + c1['close']) / 2 and
            c2['close'] < c1['open']):
            patterns['PIERCING_LINE'] = 'bullish'

        # Dark Cloud Cover
        if (body_color(c1['open'], c1['close']) == 'green' and
            body_color(c2['open'], c2['close']) == 'red' and
            c2['open'] > c1['close'] and
            c2['close'] < (c1['open'] + c1['close']) / 2 and
            c2['close'] > c1['open']):
            patterns['DARK_CLOUD_COVER'] = 'bearish'

        # Tweezer Bottoms
        if (c1['low'] == c2['low'] and
            body_color(c1['open'], c1['close']) == 'red' and
            body_color(c2['open'], c2['close']) == 'green'):
            patterns['TWEEZER_BOTTOM'] = 'bullish'

        # Tweezer Tops
        if (c1['high'] == c2['high'] and
            body_color(c1['open'], c1['close']) == 'green' and
            body_color(c2['open'], c2['close']) == 'red'):
            patterns['TWEEZER_TOP'] = 'bearish'

        # Bullish Kicker
        if (body_color(c1['open'], c1['close']) == 'red' and
            body_color(c2['open'], c2['close']) == 'green' and
            c2['open'] > c1['close']):
            patterns['BULLISH_KICKER'] = 'bullish'

        # Bearish Kicker
        if (body_color(c1['open'], c1['close']) == 'green' and
            body_color(c2['open'], c2['close']) == 'red' and
            c2['open'] < c1['close']):
            patterns['BEARISH_KICKER'] = 'bearish'

    except Exception as e:
        print(f"⚠️ Manual pattern detection error: {e}")

    return patterns

def get_pattern_signal(patterns):
    bullish = sum(1 for p, d in patterns.items() if d == 'bullish')
    bearish = sum(1 for p, d in patterns.items() if d == 'bearish')
    neutral = sum(1 for p, d in patterns.items() if d == 'neutral')
    total = bullish + bearish + neutral
    if total == 0:
        return 'HOLD', 0.0
    if bullish > bearish:
        signal = 'BUY'
        confidence = bullish / (bullish + bearish + 1e-6)
    elif bearish > bullish:
        signal = 'SELL'
        confidence = bearish / (bullish + bearish + 1e-6)
    else:
        signal = 'HOLD'
        confidence = 0.0
    return signal, confidence