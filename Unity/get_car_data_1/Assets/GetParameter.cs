using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class GetParameter : MonoBehaviour
    {
        public float myaccel = 0.0f;
        // Use this for initialization
        void Start()
        {
          
    }

        // Update is called once per frame
        void Update()
        {
            this.GetComponent<Text>().text = "v" + myaccel.ToString();
        }
    }
