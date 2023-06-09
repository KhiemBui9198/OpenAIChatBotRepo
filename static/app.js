class Chatbox {
    constructor() {
        this.args = {
            openButton: document.querySelector('.chatbox__button'),
            chatBox: document.querySelector('.chatbox__support'),
            sendButton: document.querySelector('.send__button')
        }
        this.state = false;
        this.messages = [];
    }

    display() {
        this.hideElem();
        const { openButton, chatBox, sendButton } = this.args;
        openButton.addEventListener('click', () => this.toggleState(chatBox))

        sendButton.addEventListener('click', () => this.onSendButton(chatBox))
        const node = chatBox.querySelector('input');
        node.addEventListener("keyup", ({ key }) => {
            if (key === "Enter") {
                this.onSendButton(chatBox)
            }
        })
    }

    toggleState(chatbox) {
        this.state = !this.state;
        // show or hides the box
        if (this.state) {
            chatbox.classList.add('chatbox--active')
            if (this.isfirst == null) {
                let msg2 = { name: "Sam", message: 'Xin chào tôi là chat bot tư vấn, tôi có thể giúp gì cho bạn ?' };
                this.messages.push(msg2);
                this.updateChatText(chatbox)
            }
            this.isfirst = false;
        } else {
            chatbox.classList.remove('chatbox--active')
        }
    }
    hideElem() {
        document.getElementById("myTyping").style.visibility = "hidden";
      }
      
    showElem() {
        document.getElementById("myTyping").style.visibility = "visible";
      }
    onSendButton(chatbox) {
        this.showElem();
        var textField = chatbox.querySelector('input');
        let text1 = textField.value
        if (text1 === "") {
            return;
        }textField.value = ' ';
        let msg1 = { name: "User", message: text1 }
        this.messages.push(msg1);
        fetch('/predict', {
            method: 'POST',
            body: JSON.stringify({ message: text1 }),
            mode: 'cors',
            headers: {
                'Content-Type': 'application/json'
            },
        })
            .then(r => r.json())
            .then(r => {
                let msg2 = { name: "Sam", message: r.answer };
                this.messages.push(msg2);
                this.updateChatText(chatbox)
                textField.value = '';
                this.hideElem();

            }).catch((error) => {
                console.error('Error:', error);
                this.updateChatText(chatbox)
                textField.value = '';
                this.hideElem();
            });
    }

    updateChatText(chatbox) {
        var html = '';
        this.messages.slice().reverse().forEach(function (item, index) {
            if (item.name === "Sam") {
                html += '<div class="messages__item messages__item--visitor">' + item.message + '</div>'
            }
            else {
                html += '<div class="messages__item messages__item--operator">' + item.message + '</div>'
            }
        });
        const chatmessage = chatbox.querySelector('.chatbox__messages');
        chatmessage.innerHTML = html;
    }

}
const chatbox = new Chatbox();
chatbox.display();


