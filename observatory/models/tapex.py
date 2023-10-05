def tapex_inference(model, batch_input_ids, batch_attention_mask):
    # Retrieve encoder's output
    encoder_outputs = model.model.encoder(batch_input_ids, attention_mask=batch_attention_mask)

    # Retrieve decoder's output using encoder's outputs and attention mask
    decoder_outputs = model.model.decoder(
        input_ids=batch_input_ids, 
        encoder_hidden_states=encoder_outputs[0], 
        attention_mask=batch_attention_mask
    )

    # The first output of the decoder contains the last hidden states
    return decoder_outputs[0]