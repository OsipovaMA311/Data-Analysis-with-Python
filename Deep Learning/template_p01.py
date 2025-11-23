import numpy as np

def softmax(vector):
    '''
    vector: np.array of shape (n, m)
    
    return: np.array of shape (n, m)
        Matrix where softmax is computed for every row independently
    '''
    nice_vector = vector - vector.max()
    exp_vector = np.exp(nice_vector)
    exp_denominator = np.sum(exp_vector, axis=1)[:, np.newaxis]
    softmax_ = exp_vector / exp_denominator
    return softmax_
def multiplicative_attention(decoder_hidden_state, encoder_hidden_states, W_mult):
    '''
    decoder_hidden_state: np.array of shape (n_features_dec, 1)
    encoder_hidden_states: np.array of shape (n_features_enc, n_states)
    W_mult: np.array of shape (n_features_dec, n_features_enc)

    return: np.array of shape (n_features_enc, 1)
        Final attention vector
    '''
    # Вычисляем attention scores: s^T * W_mult * h_i
    # decoder_hidden_state.T shape: (1, n_features_dec)
    # W_mult shape: (n_features_dec, n_features_enc)
    # encoder_hidden_states shape: (n_features_enc, n_states)
    
    # Сначала вычисляем промежуточный результат: W_mult * encoder_hidden_states
    # Результат: (n_features_dec, n_states)
    intermediate = np.dot(W_mult, encoder_hidden_states)
    
    # Затем вычисляем скалярные произведения: decoder_hidden_state.T * intermediate
    # Результат: (1, n_states)
    attention_scores = np.dot(decoder_hidden_state.T, intermediate)
    
    # Применяем softmax к attention scores
    weights_vector = softmax(attention_scores)
    
    # Вычисляем взвешенную сумму состояний энкодера
    # weights_vector shape: (1, n_states)
    # encoder_hidden_states.T shape: (n_states, n_features_enc)
    attention_vector = np.dot(weights_vector, encoder_hidden_states.T).T
    
    return attention_vector

def additive_attention(decoder_hidden_state, encoder_hidden_states, v_add, W_add_enc, W_add_dec):
    '''
    decoder_hidden_state: np.array of shape (n_features_dec, 1)
    encoder_hidden_states: np.array of shape (n_features_enc, n_states)
    v_add: np.array of shape (n_features_int, 1)
    W_add_enc: np.array of shape (n_features_int, n_features_enc)
    W_add_dec: np.array of shape (n_features_int, n_features_dec)

    return: np.array of shape (n_features_enc, 1)
        Final attention vector
    '''
    # Вычисляем attention scores: v^T * tanh(W_add_enc * h_i + W_add_dec * s)
    
    # W_add_enc * encoder_hidden_states shape: (n_features_int, n_states)
    enc_part = np.dot(W_add_enc, encoder_hidden_states)
    
    # W_add_dec * decoder_hidden_state shape: (n_features_int, 1)
    dec_part = np.dot(W_add_dec, decoder_hidden_state)
    
    # Складываем и применяем tanh
    # dec_part broadcasted to (n_features_int, n_states)
    tanh_result = np.tanh(enc_part + dec_part)
    
    # v_add.T * tanh_result shape: (1, n_states)
    attention_scores = np.dot(v_add.T, tanh_result)
    
    # Применяем softmax к attention scores
    weights_vector = softmax(attention_scores)
    
    # Вычисляем взвешенную сумму состояний энкодера
    # weights_vector shape: (1, n_states)
    # encoder_hidden_states.T shape: (n_states, n_features_enc)
    attention_vector = np.dot(weights_vector, encoder_hidden_states.T).T
    
    return attention_vector
# def multiplicative_attention(decoder_hidden_state, encoder_hidden_states, W_mult):
#     '''
#     decoder_hidden_state: np.array of shape (n_features_dec, 1)
#     encoder_hidden_states: np.array of shape (n_features_enc, n_states)
#     W_mult: np.array of shape (n_features_dec, n_features_enc)
    
#     return: np.array of shape (n_features_enc, 1)
#         Final attention vector
#     '''
#     # your code here
    
#     return attention_vector

# def additive_attention(decoder_hidden_state, encoder_hidden_states, v_add, W_add_enc, W_add_dec):
#     '''
#     decoder_hidden_state: np.array of shape (n_features_dec, 1)
#     encoder_hidden_states: np.array of shape (n_features_enc, n_states)
#     v_add: np.array of shape (n_features_int, 1)
#     W_add_enc: np.array of shape (n_features_int, n_features_enc)
#     W_add_dec: np.array of shape (n_features_int, n_features_dec)
    
#     return: np.array of shape (n_features_enc, 1)
#         Final attention vector
#     '''
#     # your code here
    
#     return attention_vector
